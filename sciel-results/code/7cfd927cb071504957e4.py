import time
import numpy as np
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin

    # Precompute all 4-cycles for verification
    all_cycles = []
    vertices = list(range(N))
    
    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        all_cycles.append((edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a)))
        # Cycle 2: a-b-d-c-a
        all_cycles.append((edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a)))
        # Cycle 3: a-c-b-d-a
        all_cycles.append((edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a)))
        
    def is_fully_valid(colors):
        # Vectorized check
        for cycle in all_cycles:
            c0 = colors[cycle[0]]
            c1 = colors[cycle[1]]
            c2 = colors[cycle[2]]
            c3 = colors[cycle[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    best_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    best_score = 0
    rng = random.Random(42)

    # Strategy: Vertex Pairing Decomposition
    # 1. Partition 16 vertices into 8 pairs.
    # 2. Assign a unique color to each of the 8 intra-pair edges.
    # 3. For inter-pair edges, we need to assign colors such that no 4-cycle has 4 distinct colors.
    #    A 4-cycle involves 4 vertices. Let's analyze the structure of a 4-cycle in terms of pairs.
    #    Case 1: The 4 vertices come from 4 distinct pairs.
    #            Edges in the cycle are all inter-pair.
    #            To avoid 4 distinct colors, we need to restrict the colors of inter-pair edges.
    #            If we assign the same color to ALL inter-pair edges, we use 8+1=9 colors.
    #            If we assign colors based on the pair indices, e.g., color(i,j) = (pair_i + pair_j) % K,
    #            does this help?
    #            Let's try a simpler structural constraint first.
    #            What if we assign inter-pair edges colors from a small set, say {8, 9, 10}?
    #            If we can use 3 colors for inter-pair edges, total = 8 + 3 = 11.
    #            Can we use more?
    #            Let's try to use the pair index to determine the color.
    #            Let pairs be P0, P1, ..., P7.
    #            Edge between Pi and Pj gets color 8 + (i+j) % 4. This uses 4 colors for inter-pair.
    #            Total = 8 + 4 = 12.
    #            Is this valid?
    #            Consider a 4-cycle involving vertices from pairs Pi, Pj, Pk, Pl.
    #            The edges are (Pi,Pj), (Pj,Pk), (Pk,Pl), (Pl,Pi).
    #            Colors: 8+(i+j)%4, 8+(j+k)%4, 8+(k+l)%4, 8+(l+i)%4.
    #            We need these 4 to not be all distinct.
    #            This is not guaranteed. For example, if i=0, j=1, k=2, l=3.
    #            Colors: 8+1, 8+3, 8+1, 8+3 -> {9, 11, 9, 11}. Not 4 distinct. Valid.
    #            Try i=0, j=1, k=2, l=4.
    #            Colors: 8+1, 8+3, 8+6%4=2, 8+4%4=0 -> {9, 11, 10, 8}. 4 distinct! INVALID.
    #            
    #            So simple modular arithmetic on pair indices doesn't work directly for 4 distinct pairs.
    #            
    #            Let's rethink.
    #            If we use only 2 colors for inter-pair edges, say 8 and 9.
    #            Then any 4-cycle with 4 distinct pairs has colors from {8,9} for inter-pair edges.
    #            Since there are only 2 colors, we can't have 4 distinct colors.
    #            What about cycles involving intra-pair edges?
    #            A 4-cycle can involve at most 2 intra-pair edges (if it uses 2 pairs and 2 vertices from each? No, a pair has only 2 vertices.
    #            If a cycle uses vertices from pair Pi, it can use at most 2 vertices from Pi.
    #            If it uses both vertices from Pi, the edge between them is intra-pair.
    #            Can a 4-cycle have 2 intra-pair edges?
    #            Vertices: a,b in Pi; c,d in Pj.
    #            Cycle: a-c-b-d-a.
    #            Edges: (a,c) inter, (c,b) inter, (b,d) inter, (d,a) inter. No intra-pair edges in the cycle itself, unless the cycle is a-b-...-a?
    #            Wait, the cycle is a sequence of 4 distinct vertices.
    #            If the vertices are a,b from Pi and c,d from Pj.
    #            Possible cycles:
    #            1. a-b-c-d-a: Edges (a,b) intra, (b,c) inter, (c,d) intra, (d,a) inter.
    #               Colors: C_intra_i, C_inter_bc, C_intra_j, C_inter_da.
    #               If C_intra_i, C_intra_j, C_inter_bc, C_inter_da are all distinct, it's invalid.
    #               We have 8 intra colors (unique per pair) and 2 inter colors (8,9).
    #               C_intra_i is one of 0..7. C_intra_j is one of 0..7 (different from i).
    #               C_inter_bc is 8 or 9. C_inter_da is 8 or 9.
    #               Since intra colors are 0..7 and inter are 8,9, they are disjoint sets.
    #               So we have 4 values: x, y, z, w where x!=y (intra), z,w in {8,9}.
    #               For 4 distinct, we need z != w.
    #               So if z and w are different, we have 4 distinct colors.
    #               We need to ensure that for any such cycle, the inter-pair edges (b,c) and (d,a) do NOT have distinct colors.
    #               i.e., color(b,c) == color(d,a).
    #               This implies a constraint on the inter-pair coloring.
    #               If we use only 1 color for all inter-pair edges, this is satisfied.
    #               Total colors: 8 + 1 = 9.
    #               
    #            Can we do better?
    #            What if we use 2 inter colors, but ensure that for any two pairs Pi, Pj, the edges between them form a bipartite graph K2,2 colored with 2 colors such that opposite edges have the same color?
    #            i.e., color(a,c) = color(b,d) and color(a,d) = color(b,c).
    #            This is a "Latin square" like structure or just a checkerboard.
    #            If we assign colors to inter-pair edges based on the parity of the vertex index within the pair?
    #            Let Pi = {2i, 2i+1}.
    #            Let color(u,v) for u in Pi, v in Pj be determined by (u%2 + v%2) % 2.
    #            If u%2=0, v%2=0 -> 0. u%2=1, v%2=1 -> 0.
    #            u%2=0, v%2=1 -> 1. u%2=1, v%2=0 -> 1.
    #            So edges between same-parity vertices get color 8, different-parity get color 9.
    #            Let's check the cycle a-b-c-d-a with a,b in Pi, c,d in Pj.
    #            Assume a=2i, b=2i+1. c=2j, d=2j+1.
    #            Edges:
    #            (a,b): intra, color Ci.
    #            (b,c): b=2i+1 (odd), c=2j (even). Diff parity -> Color 9.
    #            (c,d): intra, color Cj.
    #            (d,a): d=2j+1 (odd), a=2i (even). Diff parity -> Color 9.
    #            Colors: Ci, 9, Cj, 9.
    #            Distinct values: {Ci, Cj, 9}. At most 3 distinct. Valid!
    #            
    #            What about other cycles?
    #            Cycle a-c-b-d-a.
    #            (a,c): a even, c even -> Same parity -> Color 8.
    #            (c,b): c even, b odd -> Diff parity -> Color 9.
    #            (b,d): b odd, d odd -> Same parity -> Color 8.
    #            (d,a): d odd, a even -> Diff parity -> Color 9.
    #            Colors: 8, 9, 8, 9. At most 2 distinct. Valid!
    #            
    #            What about cycles involving 3 or 4 pairs?
    #            If 3 pairs: e.g., a,b in Pi; c in Pj; d in Pk.
    #            Cycle a-b-c-d-a.
    #            (a,b) intra Ci.
    #            (b,c) inter.
    #            (c,d) inter.
    #            (d,a) inter.
    #            Colors: Ci, C_bc, C_cd, C_da.
    #            Ci is 0..7. Others are 8 or 9.
    #            Max distinct: 3 (if C_bc != C_cd and C_da is one of them, or all 3 inter are distinct? No, inter are only 8,9).
    #            So inter colors are from {8,9}.
    #            The set is {Ci, x, y, z} where x,y,z in {8,9}.
    #            Distinct count <= 1 (for Ci) + 2 (for inter) = 3. Valid.
    #            
    #            If 4 pairs: a in Pi, b in Pj, c in Pk, d in Pl.
    #            All edges inter.
    #            Colors from {8,9}.
    #            Distinct count <= 2. Valid.
    #            
    #            So this strategy guarantees validity!
    #            Total colors: 8 (intra) + 2 (inter) = 10.
    #            Current best is 16. We need to do better.
    #            
    #            How to increase colors?
    #            The bottleneck is the inter-pair edges. We used 2 colors.
    #            Can we use more colors for inter-pair edges while maintaining the property?
    #            The constraint was: for any 4 vertices from 2 pairs, the 4 inter edges in the two possible cycles must not create 4 distinct colors.
    #            Actually, the critical constraint was for the cycle a-b-c-d-a where a,b in Pi, c,d in Pj.
    #            We needed color(b,c) and color(d,a) to not be distinct IF Ci and Cj are distinct from them and each other.
    #            Wait, in our previous analysis:
    #            Colors: Ci, C_bc, Cj, C_da.
    #            We need NOT (Ci != Cj != C_bc != C_da != Ci ...).
    #            Specifically, we need at least one equality among these 4.
    #            Since Ci != Cj always (unique intra colors), we need C_bc == C_da OR C_bc == Ci OR C_da == Cj OR C_bc == Cj OR C_da == Ci.
    #            Since Ci, Cj are 0..7 and C_bc, C_da are 8.., the equalities with intra colors are impossible if we keep inter colors >= 8.
    #            So we MUST have C_bc == C_da.
    #            This means for any two pairs Pi, Pj, and any assignment of vertices a,b to Pi and c,d to Pj,
    #            the color of edge (b,c) must equal the color of edge (d,a).
    #            Note: a,b are the two vertices in Pi. c,d are the two vertices in Pj.
    #            This implies that the 4 edges between Pi and Pj must be colored such that:
    #            color(2i, 2j) = color(2i+1, 2j+1)  (Same parity)
    #            color(2i, 2j+1) = color(2i+1, 2j)  (Diff parity)
    #            And these two values can be different.
    #            So for each pair of pairs (Pi, Pj), we can choose 2 colors.
    #            However, these colors must be consistent across different pairs?
    #            No, the constraint is local to the pair of pairs.
    #            BUT, we also have constraints from 4-cycles involving 3 or 4 pairs.
    #            
    #            Let's check the 4-pair case again with this flexibility.
    #            Vertices a in Pi, b in Pj, c in Pk, d in Pl.
    #            Cycle a-b-c-d-a.
    #            Edges: (a,b) from Pi,Pj. (b,c) from Pj,Pk. (c,d) from Pk,Pl. (d,a) from Pl,Pi.
    #            Let C_PiPj(a,b) be the color.
    #            We need these 4 colors to not be all distinct.
    #            
    #            If we assign colors to inter-pair edges based on the pair indices ONLY, e.g., C(Pi, Pj) = f(i,j),
    #            then for a 4-cycle involving 4 distinct pairs, the colors are f(i,j), f(j,k), f(k,l), f(l,i).
    #            We need these 4 to not be all distinct.
    #            This is a constraint on the function f defined on pairs of pairs.
    #            There are 8 pairs of pairs. So f is a function from C(8,2) to Colors.
    #            We want to maximize the number of distinct values in f, subject to:
    #            For any 4 distinct pairs i,j,k,l, the values f(i,j), f(j,k), f(k,l), f(l,i) are not all distinct.
    #            Also, we have the "2-pair" constraint which was satisfied by the parity structure, but if we use a single color per pair-of-pairs, the parity structure is lost.
    #            If C(Pi, Pj) is a single color for all 4 edges between Pi and Pj, then:
    #            Cycle a-b-c-d-a (2 pairs): Colors Ci, C_PiPj, Cj, C_PiPj.
    #            Distinct: {Ci, Cj, C_PiPj}. Max 3. Valid.
    #            So we CAN use a single color per pair-of-pairs!
    #            
    #            So the problem reduces to:
    #            1. Assign 8 unique colors to intra-pair edges. (8 colors)
    #            2. Assign colors to the 28 inter-pair connections (between the 8 pairs) such that:
    #               For any 4 distinct pairs i,j,k,l, the colors of (i,j), (j,k), (k,l), (l,i) are not all distinct.
    #            Let G be the complete graph K8 on the 8 pairs. We are coloring the edges of K8.
    #            Constraint: No 4-cycle in K8 has 4 distinct edge colors.
    #            This is exactly the same problem but with n=8 instead of n=16!
    #            And we want to maximize the number of colors used in this K8 coloring.
    #            Plus the 8 intra colors.
    #            
    #            So, Score = 8 + (Max colors in valid K8 coloring).
    #            
    #            Let's find the max colors for K8.
    #            If we can find a valid K8 coloring with C8 colors, total = 8 + C8.
    #            Current best is 16. So we need C8 >= 8.
    #            Can we color K8 with 8 colors such that no 4-cycle has 4 distinct colors?
    #            If we use 8 colors, that's 1 per edge? No, 28 edges.
    #            Let's try to construct a K8 coloring.
    #            
    #            Heuristic for K8:
    #            Try to use as many colors as possible.
    #            Let's run a local search for K8.
    #            
    #            If we find a K8 coloring with, say, 10 colors, total = 18.
    #            If 12 colors, total = 20.
    #            
    #            Let's implement a solver for the K8 subproblem.
    #            N_sub = 8. Edges = 28.
    #            Cycles in K8: C(8,4) * 3 = 70 * 3 = 210 cycles.
    #            This is small enough for fast search.
    #            
    #            We can precompute the best K8 coloring offline or at runtime.
    #            Since 28 edges is small, we can do a quick greedy or local search.
    #            
    #            Let's write a function to find a good K8 coloring.
    
    def solve_k8():
        n = 8
        num_edges = 28
        # Edge indices in K8
        def e_idx(i, j):
            if i > j: i, j = j, i
            return i * n - i * (i + 1) // 2 + (j - i - 1)
            
        # Precompute cycles
        cycles = []
        for quad in combinations(range(n), 4):
            a,b,c,d = quad
            cycles.append((e_idx(a,b), e_idx(b,c), e_idx(c,d), e_idx(d,a)))
            cycles.append((e_idx(a,b), e_idx(b,d), e_idx(d,c), e_idx(c,a)))
            cycles.append((e_idx(a,c), e_idx(c,b), e_idx(b,d), e_idx(d,a)))
            
        def is_valid_k8(colors):
            for c in cycles:
                c0,c1,c2,c3 = colors[c[0]], colors[c[1]], colors[c[2]], colors[c[3]]
                if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                    return False
            return True
            
        best = np.zeros(num_edges, dtype=np.int32)
        best_score = 0
        rng = random.Random(123)
        
        # Greedy initialization: try to use new colors
        # Order edges randomly
        edge_order = list(range(num_edges))
        rng.shuffle(edge_order)
        
        for _ in range(1000): # 1000 restarts
            if time.time() > deadline - 5: break
            colors = np.zeros(num_edges, dtype=np.int32)
            used = set([0])
            
            for e in edge_order:
                # Try to find a new color
                found = False
                # Try random new colors
                for _ in range(20):
                    c = rng.randint(0, 119)
                    if c not in used:
                        # Check validity
                        valid = True
                        for cyc in cycles:
                            if e in cyc:
                                pos = cyc.index(e)
                                others = [cyc[i] for i in range(4) if i != pos]
                                c1,c2,c3 = colors[others[0]], colors[others[1]], colors[others[2]]
                                if c != c1 and c != c2 and c != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                                    valid = False
                                    break
                        if valid:
                            colors[e] = c
                            used.add(c)
                            found = True
                            break
                if not found:
                    # Try existing colors
                    for c in list(used):
                        valid = True
                        for cyc in cycles:
                            if e in cyc:
                                pos = cyc.index(e)
                                others = [cyc[i] for i in range(4) if i != pos]
                                c1,c2,c3 = colors[others[0]], colors[others[1]], colors[others[2]]
                                if c != c1 and c != c2 and c != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                                    valid = False
                                    break
                        if valid:
                            colors[e] = c
                            found = True
                            break
                if not found:
                    colors[e] = 0 # Fallback
            
            if is_valid_k8(colors):
                sc = len(set(colors))
                if sc > best_score:
                    best_score = sc
                    best = colors.copy()
                    
            # Local search improvement
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for _ in range(100):
                    e = rng.randint(0, num_edges-1)
                    old_c = colors[e]
                    # Try to change to a new color
                    new_c = rng.randint(0, 119)
                    if new_c == old_c or new_c in set(colors):
                        continue
                    # Check validity
                    valid = True
                    for cyc in cycles:
                        if e in cyc:
                            pos = cyc.index(e)
                            others = [cyc[i] for i in range(4) if i != pos]
                            c1,c2,c3 = colors[others[0]], colors[others[1]], colors[others[2]]
                            if new_c != c1 and new_c != c2 and new_c != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                                valid = False
                                break
                    if valid:
                        colors[e] = new_c
                        improved = True
                        sc = len(set(colors))
                        if sc > best_score:
                            best_score = sc
                            best = colors.copy()
                            
        return best, best_score

    # Solve the K8 subproblem
    k8_colors, k8_score = solve_k8()
    
    # Construct the full K16 coloring
    # Pairs: (0,1), (2,3), ..., (14,15)
    # Intra-pair edges get colors 0..7
    # Inter-pair edges get colors 8 + k8_color
    
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    
    # Assign intra-pair colors
    for i in range(8):
        u = 2*i
        v = 2*i+1
        idx = edge_index(u, v)
        colors[idx] = i
        
    # Assign inter-pair colors
    # Map K8 edge index to K16 inter-pair edges
    # K8 vertices 0..7 correspond to K16 pairs 0..7
    # K8 edge (i,j) corresponds to all 4 edges between pair i and pair j in K16
    
    # Precompute K8 edge index function
    def k8_edge_index(i, j):
        if i > j: i, j = j, i
        return i * 8 - i * (i + 1) // 2 + (j - i - 1)
        
    for i in range(8):
        for j in range(i+1, 8):
            k8_idx = k8_edge_index(i, j)
            c_val = 8 + k8_colors[k8_idx]
            
            # Get the 4 vertices
            u1, u2 = 2*i, 2*i+1
            v1, v2 = 2*j, 2*j+1
            
            # Edges: (u1,v1), (u1,v2), (u2,v1), (u2,v2)
            for u in [u1, u2]:
                for v in [v1, v2]:
                    idx = edge_index(u, v)
                    colors[idx] = c_val
                    
    # Verify and count
    if is_fully_valid(colors):
        score = len(set(colors))
        if score > best_score:
            best_score = score
            best_colors = colors.copy()
    else:
        # Should not happen if logic is correct, but fallback
        best_colors = np.zeros(NUM_EDGES, dtype=np.int32)
        best_score = 1

    # Try to improve the K16 coloring directly?
    # The structure is rigid. Improving K8 improves K16.
    # We already did local search on K8.
    
    # One more check: Can we mix intra/inter colors?
    # No, intra are 0-7, inter are 8+. Disjoint.
    
    return best_colors.tolist()

if __name__ == "__main__":
    colors = construct()
    print(f"Max colors: {len(set(colors))}")
