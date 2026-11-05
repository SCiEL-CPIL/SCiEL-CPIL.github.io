import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    def check_violation(n, colors):
        """
        Returns None if valid.
        Returns a list of 6 vertices forming a monochromatic K2,2,2 if invalid.
        """
        mask0 = [0] * n
        mask1 = [0] * n
        
        # Build masks
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[k]
                if c == 0:
                    mask0[i] |= (1 << j)
                    mask0[j] |= (1 << i)
                else:
                    mask1[i] |= (1 << j)
                    mask1[j] |= (1 << i)
                k += 1
        
        def check_color(mask):
            # Iterate over pairs (a,b) and (c,d)
            # To find K2,2,2:
            # 1. Find a K2,2 (4-cycle) in the graph.
            # 2. Check if the 4 vertices have at least 2 common neighbors.
            
            # Optimization: Iterate over 4-tuples (a,b,c,d) with a<b<c<d.
            # Check if (a,b), (c,d), (a,c), (a,d), (b,c), (b,d) are all in mask.
            # This forms a K4 minus two edges? No.
            # K2,2 on {a,b} and {c,d} requires edges:
            # (a,c), (a,d), (b,c), (b,d).
            # The edges (a,b) and (c,d) are NOT required for K2,2.
            # However, the problem statement says "monochromatic octahedron K2,2,2".
            # K2,2,2 is the complete tripartite graph with parts of size 2.
            # Let parts be A={a1,a2}, B={b1,b2}, C={c1,c2}.
            # Edges exist between all pairs in different parts.
            # No edges within a part.
            # So, for vertices v1..v6, we need to partition them into 3 pairs such that
            # all 12 cross-edges are present in the monochromatic graph.
            
            # My previous check logic in the seed was:
            # "Find (a,b) and (c,d) such that (a,b) is edge, (c,d) is edge, and (a,c),(a,d),(b,c),(b,d) are edges."
            # This forms a K4 on {a,b,c,d}.
            # Then "common neighbors of a,b,c,d".
            # If e,f are common neighbors, then e,f are connected to a,b,c,d.
            # Are e,f connected to each other? Not required for K2,2,2 structure if e,f are in the same part.
            # Are a,b connected? Yes (edge).
            # Are c,d connected? Yes (edge).
            # Are a,c connected? Yes.
            # This forms a structure where {a,b}, {c,d}, {e,f} are the parts?
            # Let's verify:
            # Part 1: {a,b}. Edge (a,b) present.
            # Part 2: {c,d}. Edge (c,d) present.
            # Part 3: {e,f}. Edge (e,f) NOT required to be present.
            # Cross edges:
            # a-c, a-d, b-c, b-d (Present by K4 check)
            # a-e, a-f, b-e, b-f, c-e, c-f, d-e, d-f (Present by common neighbor check)
            # This IS a K2,2,2.
            
            # So the check is:
            # Find 4 vertices a,b,c,d such that all 6 edges of K4 are present?
            # No, the seed code checked:
            # if (mask[a] & (1<<b)) # Edge (a,b)
            # if (mask[c] & (1<<d)) # Edge (c,d)
            # if (mask[a] & (1<<c)) and ... # Cross edges
            # This ensures {a,b,c,d} form a K4.
            # Then common = intersection of neighbors of a,b,c,d.
            # If common has >= 2 bits, we have e,f.
            # This confirms K2,2,2.
            
            # Is it possible to have K2,2,2 without a K4?
            # Parts A={a1,a2}, B={b1,b2}, C={c1,c2}.
            # Edges: all between parts.
            # If we pick a1, a2 from A, b1, b2 from B.
            # Edges (a1,b1), (a1,b2), (a2,b1), (a2,b2) are present.
            # Edges (a1,a2) NOT present (same part).
            # Edges (b1,b2) NOT present (same part).
            # So {a1,a2,b1,b2} does NOT form a K4. It forms a K2,2 (4-cycle with no diagonals).
            
            # The seed code's check:
            # 1. (a,b) is edge.
            # 2. (c,d) is edge.
            # 3. Cross edges present.
            # This implies (a,b) and (c,d) are edges.
            # In a K2,2,2, can we find 4 vertices that form a K4?
            # Pick one from each part? No, K4 needs 4 vertices.
            # Pick 2 from A, 2 from B? No edges within A or B.
            # Pick 2 from A, 1 from B, 1 from C?
            # Let a1,a2 in A. b1 in B. c1 in C.
            # Edges:
            # a1-a2: No.
            # So no K4 exists in a K2,2,2 graph!
            
            # WAIT. The seed code's check is WRONG for K2,2,2?
            # Let's re-read the task: "Neither colour class may contain K2,2,2 as a (not necessarily induced) subgraph."
            # K2,2,2 is the complete tripartite graph K_{2,2,2}.
            # It has 12 edges.
            # Does it contain a K4?
            # Vertices: a1,a2 | b1,b2 | c1,c2.
            # Any 4 vertices?
            # 2 from A, 2 from B: No edges within A or B. Induced subgraph is C4. Not K4.
            # 2 from A, 1 from B, 1 from C:
            # a1-a2: No.
            # So no 4 vertices form a K4.
            
            # Therefore, the seed program's check `check_violation` is INCORRECT.
            # It was checking for a K4 with 2 common neighbors.
            # A K4 with 2 common neighbors e,f forms a K2,2,2?
            # Vertices a,b,c,d (K4) and e,f (common neighbors).
            # Edges:
            # a,b,c,d form K4.
            # e,f connected to all a,b,c,d.
            # Is e-f connected? Not required.
            # Does this graph contain K2,2,2?
            # Let's try to map {a,b,c,d,e,f} to parts A,B,C.
            # If we map {a,b} to A, {c,d} to B, {e,f} to C.
            # Edges within A: a-b is present (K4). But K2,2,2 requires NO edges within parts.
            # So this configuration is NOT a valid embedding of K2,2,2 if we assume the parts are disjoint sets with no internal edges.
            # BUT, the subgraph doesn't have to be induced.
            # "Not necessarily induced" means we just need the edges of K2,2,2 to be present.
            # If we have a K4 on a,b,c,d, and e,f connected to all, do we have K2,2,2?
            # We need to partition 6 vertices into 3 pairs such that all cross-edges exist.
            # If we pick pairs (a,b), (c,d), (e,f).
            # Cross edges between (a,b) and (c,d): a-c, a-d, b-c, b-d. All present (K4).
            # Cross edges between (a,b) and (e,f): a-e, a-f, b-e, b-f. All present (common neighbors).
            # Cross edges between (c,d) and (e,f): c-e, c-f, d-e, d-f. All present.
            # So YES, the existence of a K4 with 2 common neighbors IMPLIES the existence of K2,2,2.
            # (The extra edges a-b, c-d, e-f (if present) don't hurt, since it's not induced).
            
            # However, is it possible to have K2,2,2 WITHOUT a K4?
            # As analyzed before, K2,2,2 itself has no K4.
            # But the graph is "not necessarily induced".
            # If the graph contains K2,2,2 as a subgraph, it contains the 12 edges.
            # Does it necessarily contain a K4?
            # No. The K2,2,2 subgraph itself has no K4.
            # But the rest of the graph might have edges that create a K4.
            # If the graph is EXACTLY K2,2,2 (plus isolated vertices), there is no K4.
            # So the seed's check MISSES valid K2,2,2 violations that don't contain a K4!
            
            # Example: The graph IS K2,2,2.
            # Vertices 0,1 in A. 2,3 in B. 4,5 in C.
            # Edges: all cross.
            # Check for K4:
            # Any 4 vertices?
            # 0,1,2,3: Edges 0-2, 0-3, 1-2, 1-3. Missing 0-1, 2-3. Not K4.
            # So seed check returns None (Valid).
            # But it is INVALID (contains K2,2,2).
            
            # So the seed program is buggy and likely returns invalid colorings for n=29 or higher if the specific random walk didn't happen to create K4s.
            # OR, the random walk for n=29 happened to find a coloring that avoids K4s with common neighbors, but might still have K2,2,2s?
            # Actually, if it returns a coloring, the evaluator checks it.
            # The evaluator likely checks for K2,2,2 correctly.
            # If the seed's internal check is weak, it might return a coloring that the evaluator deems invalid.
            # BUT the score is 1.0, which corresponds to n=29 (record).
            # This implies the seed's output for n=29 was accepted by the evaluator.
            # This means the specific coloring found by the seed did NOT contain K2,2,2.
            # But the check itself is flawed.
            
            # For my new strategy, I MUST implement a CORRECT check.
            
            # Correct Check for K2,2,2:
            # Iterate over all 6-subsets? O(n^6) is too slow.
            # Iterate over partitions of 6 vertices into 3 pairs?
            # Number of ways to partition 6 items into 3 pairs is 15.
            # For each 6-subset, check 15 pairings.
            # Number of 6-subsets in n=30 is ~593,775.
            # 593,775 * 15 * (12 edge checks) ~ 100 million operations.
            # In Python, this is too slow (seconds per check).
            
            # Alternative:
            # Iterate over pairs of pairs (A,B) and check common neighbors?
            # Let A={a1,a2}, B={b1,b2}.
            # If all 4 cross edges exist, then A and B form a "K2,2 block".
            # Let C be the set of common neighbors of a1,a2,b1,b2.
            # If |C| >= 2, and we pick c1,c2 from C, do we have K2,2,2?
            # We need edges between C and A, and C and B.
            # Since c1,c2 are common neighbors, they are connected to a1,a2,b1,b2.
            # So edges c1-a1, c1-a2, c1-b1, c1-b2 exist.
            # Same for c2.
            # We also need edges between A and B (exist by definition of K2,2 block).
            # Do we need edges within A, B, C? No.
            # So, if we find two disjoint pairs A,B that form a K2,2, and their common neighbor set C has size >= 2,
            # then A, B, and any two vertices from C form a K2,2,2.
            
            # This is the same logic as the seed, BUT the seed required A and B to be EDGES (a1-a2, b1-b2).
            # In K2,2,2, pairs in the same part do NOT have edges.
            # So the seed's requirement "mask[a] & (1<<b)" is WRONG.
            # It should allow a1,a2 to be non-adjacent.
            
            # Corrected Logic:
            # Iterate over 4 vertices a,b,c,d.
            # Check if they form a K2,2 (4-cycle).
            # i.e., (a-c), (a-d), (b-c), (b-d) are edges.
            # (a-b) and (c-d) can be anything.
            # If they form K2,2, find common neighbors of a,b,c,d.
            # If common neighbors >= 2, we have K2,2,2.
            
            # Complexity:
            # 4-tuples: O(n^4) ~ 800k for n=30.
            # Each check: 4 ANDs for K2,2, 1 AND for common.
            # This is fast enough.
            
            for a in range(n):
                for b in range(a + 1, n):
                    for c in range(b + 1, n):
                        # Check if (a,c), (a,d), (b,c), (b,d) are edges
                        # We need to iterate d?
                        # To ensure disjointness and ordering:
                        # We are looking for 4 vertices.
                        # Let's just iterate 4-tuples a<b<c<d.
                        # Check if (a,c), (a,d), (b,c), (b,d) are edges.
                        # This forms K2,2 on {a,b} and {c,d}.
                        # Note: a,b are not necessarily connected. c,d are not necessarily connected.
                        if (mask[a] & (1 << c)) and (mask[a] & (1 << d)) and \
                           (mask[b] & (1 << c)) and (mask[b] & (1 << d)):
                            common = mask[a] & mask[b] & mask[c] & mask[d]
                            # Remove a,b,c,d from common?
                            # mask[x] does not contain x.
                            # But mask[a] contains c,d.
                            # mask[c] contains a,d.
                            # Intersection will contain a?
                            # mask[a] has c,d. mask[b] has c,d. mask[c] has a,b. mask[d] has a,b.
                            # mask[a] & mask[b] contains c,d.
                            # mask[c] & mask[d] contains a,b.
                            # (mask[a] & mask[b]) & (mask[c] & mask[d]) contains (c,d) & (a,b) = 0.
                            # So common does NOT contain a,b,c,d.
                            if common & (common - 1):
                                return [a, b, c, d]
            return None

        # Check Color 0
        res = check_color(mask0)
        if res: return res
        # Check Color 1
        res = check_color(mask1)
        if res: return res
        return None

    def get_paley_coloring(p):
        """
        Generates a Paley graph coloring for prime p.
        Edge (i,j) is color 0 if (j-i) % p is a quadratic residue.
        Color 1 otherwise.
        """
        n = p
        # Find quadratic residues
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % p
                if diff in residues:
                    colors.append(0)
                else:
                    colors.append(1)
        return n, colors

    candidates = []
    
    # Candidate 1: Paley 31
    # 31 is prime, 31 % 4 = 3.
    # Paley graph on 31 vertices.
    n31, col31 = get_paley_coloring(31)
    candidates.append((31, col31))
    
    # Candidate 2: Paley 29
    # 29 is prime, 29 % 4 = 1.
    # Paley graph on 29 vertices.
    n29, col29 = get_paley_coloring(29)
    candidates.append((29, col29))
    
    # Candidate 3: Paley 31 with one vertex removed (n=30)
    # Remove vertex 0.
    # Re-index vertices 1..30 to 0..29.
    # Edges are defined by differences mod 31.
    # We need to map the edges.
    # Original vertices: 0..30.
    # New vertices: 0..29 correspond to old 1..30.
    # Edge (i, j) in new graph corresponds to (i+1, j+1) in old graph.
    # Color is determined by (j+1 - (i+1)) % 31 = (j-i) % 31.
    # So we just need to check if (j-i) % 31 is a residue.
    # Wait, the set of residues is the same.
    # So the coloring of K_30 derived from Paley 31 by removing a vertex is:
    # Color(i,j) = 0 if (j-i) % 31 in Residues(31).
    # Note: j-i can be negative? No, j>i.
    # But mod 31, j-i is 1..29.
    # This is a valid coloring for n=30.
    
    residues_31 = set((x*x) % 31 for x in range(1, 31))
    col30 = []
    for i in range(30):
        for j in range(i+1, 30):
            diff = (j - i) % 31
            if diff in residues_31:
                col30.append(0)
            else:
                col30.append(1)
    candidates.append((30, col30))

    # Candidate 4: Paley 29 with one vertex removed (n=28) - just in case
    # But we want max n.
    
    # Sort candidates by n descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    for n, colors in candidates:
        if time.time() > deadline - 5:
            break
        violation = check_violation(n, colors)
        if violation is None:
            return {'n': n, 'colors': colors}
    
    # If all fail (which would be surprising for Paley 29), fallback to random search for n=29
    # But we should trust the math.
    # If Paley 29 fails, it means it contains K2,2,2.
    # Known result: R(K2,2,2) <= 31.
    # If Paley 29 is invalid, we need a different n=29 coloring.
    # But the seed found one.
    # Let's add a small random search for n=29 as a fallback if Paley 29 fails.
    
    if time.time() < deadline - 10:
        # Try to repair Paley 29 or find a random one
        # Actually, just run a few random restarts for n=29
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            num_edges = 29 * 28 // 2
            colors = [random.randint(0, 1) for _ in range(num_edges)]
            # Simple local search
            for step in range(100):
                if time.time() > deadline - 10:
                    break
                violation = check_violation(29, colors)
                if violation is None:
                    return {'n': 29, 'colors': colors}
                # Flip a random edge in the violation
                v = violation
                random.shuffle(v)
                i, j = v[0], v[1]
                if i > j: i, j = j, i
                idx = 0
                for x in range(29):
                    for y in range(x + 1, 29):
                        if x == i and y == j:
                            break
                        idx += 1
                    else:
                        continue
                    break
                colors[idx] = 1 - colors[idx]
                
    # Final fallback: Return the best known valid structure if we have one?
    # We don't have a hardcoded valid one for n=29 in this code.
    # But the problem says "record has 29".
    # If we fail, we return n=17 (Paley 17 is definitely safe? R(K2,2,2) > 17).
    # Actually, let's just return the first candidate that passes.
    # If none pass, return a trivial valid one (n=1).
    # But we must maximize n.
    
    # If we reach here, we failed to find a valid coloring for n>=29.
    # Return n=1 to be safe? Or n=2?
    # Better to return the highest n we tested if we are unsure? No, must be valid.
    # Let's assume Paley 29 is valid.
    # If it wasn't, the loop would have continued.
    
    # Fallback: n=1
    return {'n': 1, 'colors': []}

