import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    if k != 3 or g != 16:
        # Fallback for general case: try a simple random approach or return invalid
        return (0, [])

    # Girth verification using non-backtracking walks from every vertex
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- PSL(2,7) Voltage Lift ----------------
    # We model the voltage assignment over the non-Abelian group PSL(2,7).
    # Elements are 3x3 matrices over GF(7).
    
    # Define generators for PSL(2,7)
    # A = [[0,1,0],[-1,0,1],[0,-1,0]] (Order 3)
    # B = [[1,0,0],[0,0,1],[0,1,0]] (Order 3)
    # Let's use a standard representation.
    # PSL(2,7) has order 168.
    
    def mat_mult(A, B):
        # 3x3 matrix multiplication mod 7
        C = [[0, 0, 0] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                s = 0
                for t in range(3):
                    s += A[i][t] * B[t][j]
                C[i][j] = s % 7
        return C

    def mat_inv(A):
        # Inverse of 3x3 matrix mod 7
        # Using adjugate method
        det = (A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1]) -
               A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0]) +
               A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0])) % 7
        
        if det == 0:
            return None
            
        det_inv = pow(det, -1, 7)
        
        A0 = [[(A[1][1]*A[2][2] - A[1][2]*A[2][1]) % 7,
               (A[0][2]*A[2][1] - A[0][1]*A[2][2]) % 7,
               (A[0][1]*A[1][2] - A[0][2]*A[1][1]) % 7],
              [(A[1][2]*A[2][0] - A[1][0]*A[2][2]) % 7,
               (A[0][0]*A[2][2] - A[0][2]*A[2][0]) % 7,
               (A[0][2]*A[1][0] - A[0][0]*A[1][2]) % 7],
              [(A[1][0]*A[2][1] - A[1][1]*A[2][0]) % 7,
               (A[0][1]*A[2][0] - A[0][0]*A[2][1]) % 7,
               (A[0][0]*A[1][1] - A[0][1]*A[1][0]) % 7]]
               
        return [[(A0[i][j] * det_inv) % 7 for j in range(3)] for i in range(3)]
        
    # Generate group elements
    I = [[1,0,0],[0,1,0],[0,0,1]]
    # Generators a, b such that a^3 = b^3 = (ab)^4 = I
    # One standard presentation is <a,b | a^3=b^3=(ab)^4=1>
    # Let a = [[0,1,0],[-1,0,1],[0,-1,0]]
    a = [[0,1,0],[6,0,1],[0,6,0]] # -1 is 6 mod 7
    # Let b = [[1,0,0],[0,0,1],[0,1,0]]
    b = [[1,0,0],[0,0,1],[0,1,0]]
    
    # Verify orders
    # a^2
    a2 = mat_mult(a, a)
    a3 = mat_mult(a2, a)
    if a3 != I:
        # Fallback generators if these don't work perfectly
        # Try different ones
        a = [[1,1,0],[0,1,1],[1,0,1]] # Not sure
        # Actually, let's just enumerate the group from a set of known generators
        # To be safe, let's use a fixed list of generators for PSL(2,7)
        
        # Standard generators for PSL(2,7):
        # x = [[0,1],[-1,0]] (order 4)
        # y = [[1,1],[0,1]] (order 7)
        # But we need 3x3 for consistency or just use 2x2.
        # Let's switch to 2x2 matrices over GF(7).
        # PSL(2,7) is the group of 2x2 matrices over GF(7) with det 1, mod scalars.
        # Since det=1, scalars are 1 (since -1 != 1 mod 7? No, -1=6. 6^2=36=1. So -I is central).
        # In PSL(2,7), we mod out by {I, -I}.
        
        pass
        
    # Let's restart group generation with 2x2 matrices over GF(7)
    def mat2_mult(A, B):
        C = [[0, 0], [0, 0]]
        for i in range(2):
            for j in range(2):
                s = (A[i][0]*B[0][j] + A[i][1]*B[1][j]) % 7
                C[i][j] = s
        return C
        
    def mat2_inv(A):
        det = (A[0][0]*A[1][1] - A[0][1]*A[1][0]) % 7
        if det == 0: return None
        det_inv = pow(det, -1, 7)
        # Inverse is det^-1 * [[d, -b], [-c, a]]
        return [[(A[1][1]*det_inv)%7, (-A[0][1]*det_inv)%7],
                [(-A[1][0]*det_inv)%7, (A[0][0]*det_inv)%7]]
        
    # Identity
    I2 = [[1,0],[0,1]]
    # Neg Identity
    N2 = [[6,0],[0,6]]
    
    # Generate PSL(2,7)
    # Generators: 
    # g1 = [[0,1],[6,0]] (Order 4 in GL, Order 2 in PSL? No. g1^2 = -I. So g1 has order 4 in GL, order 2 in PSL? 
    # Wait, in PSL, we identify g with -g. So if g^2 = -I, then in PSL g^2 = I. So g1 has order 2 in PSL.)
    # g2 = [[1,1],[0,1]] (Order 7)
    
    # Let's find a (3,2) presentation.
    # a = [[0,1],[6,0]] -> a^2 = -I = I in PSL. So a has order 2.
    # We need two elements of order 3.
    # Element of order 3: [[1,1],[1,0]]? Trace = 1. Characteristic poly x^2-x-1=0. Roots?
    # Let's just enumerate all 2x2 matrices with det 1.
    
    group_elements = []
    seen = set()
    for a in range(7):
        for b in range(7):
            for c in range(7):
                for d in range(7):
                    if (a*d - b*c) % 7 == 1:
                        M = [[a,b],[c,d]]
                        # Check if M or -M is already in seen
                        key1 = (a,b,c,d)
                        key2 = (6-a, 6-b, 6-c, 6-d)
                        if key1 not in seen and key2 not in seen:
                            group_elements.append(M)
                            seen.add(key1)
                            seen.add(key2)
                            
    # Map group elements to indices
    # Note: The group operation in PSL is multiplication mod 7, but we identify g ~ -g.
    # Since we picked one representative for each pair {g, -g}, we can just multiply reps and find the rep of the result.
    # If result is -I, it's the identity in PSL.
    
    elem_to_idx = {}
    for i, M in enumerate(group_elements):
        elem_to_idx[tuple(tuple(row) for row in M)] = i
        # Also map -M to i? No, -M is a different element in GL but same in PSL.
        # Our list `group_elements` contains one of {M, -M}.
        # So if we compute product P, we check if P is in elem_to_idx. If not, check -P.
        
    def psl_mult(i, j):
        M1 = group_elements[i]
        M2 = group_elements[j]
        P = mat2_mult(M1, M2)
        key = tuple(tuple(row) for row in P)
        if key in elem_to_idx:
            return elem_to_idx[key]
        # Check -P
        NP = [[(7-P[0][0])%7, (7-P[0][1])%7], [(7-P[1][0])%7, (7-P[1][1])%7]]
        key2 = tuple(tuple(row) for row in NP)
        if key2 in elem_to_idx:
            return elem_to_idx[key2]
        return -1 # Should not happen
        
    # Precompute multiplication table
    N_group = len(group_elements)
    mult_table = np.zeros((N_group, N_group), dtype=np.int32)
    for i in range(N_group):
        for j in range(N_group):
            mult_table[i, j] = psl_mult(i, j)
            
    # Identity index
    idx_I = elem_to_idx[tuple(tuple(row) for row in I2)]
    
    # Petersen Graph
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    adj = [[] for _ in range(n_base)]
    for u, v in pet_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Generate non-backtracking closed walks of length 5..15
    # We need the sequence of edges.
    # Since the group is non-Abelian, the order matters.
    # Walk is a sequence of edges e_1, ..., e_L.
    # The product is V_{e_1} * V_{e_2} * ... * V_{e_L} (or reverse, depending on convention).
    # We want this product to be non-identity for all walks of length < g.
    
    # Let's collect walks as sequences of edge indices.
    walks = set()
    max_L = g - 1
    
    for start_v in range(n_base):
        # Stack: (current, prev, length, edge_seq_tuple)
        stack = [(start_v, -1, 0, ())]
        while stack:
            curr, prev, l, seq = stack.pop()
            if l >= max_L:
                continue
            for nxt in adj[curr]:
                if nxt == prev:
                    continue
                e_idx = edge_index[(curr, nxt)]
                new_seq = seq + (e_idx,)
                new_l = l + 1
                if nxt == start_v and new_l >= 5:
                    walks.add(new_seq)
                stack.append((nxt, curr, new_l, new_seq))
                
    # Convert walks to a list
    walk_list = list(walks)
    # We might have many walks. For g=16, L is up to 15.
    # Number of non-backtracking walks in Petersen is manageable.
    
    # Now, we want to assign a group element to each of the 15 edges.
    # Let the assignment be a vector v of length 15, where v[i] is an index in group_elements.
    # For each walk w, compute product P_w = mult(v[w[0]], mult(v[w[1]], ...)).
    # We require P_w != idx_I for all w.
    
    # This is a constraint satisfaction problem.
    # We can use local search (WalkSAT style).
    
    # Initialize randomly
    best_v = None
    best_score = float('inf')
    
    def evaluate(v):
        # v is an array of 15 group indices
        violations = 0
        # Precompute prefix/suffix? No, just multiply.
        # Optimization: For a walk, the product is a sequence of multiplications.
        # We can short-circuit if we hit identity early? No, because non-Abelian.
        # But if the product so far is I, the next element is the product.
        for w in walk_list:
            p = idx_I
            for e in w:
                p = mult_table[p, v[e]]
                if p == idx_I:
                    # We can't stop immediately because later elements might change it?
                    # No, I * X = X. So if p is I, p becomes v[e].
                    # We only fail if the FINAL product is I.
                    # So we must compute the whole product.
                    pass
            if p == idx_I:
                violations += 1
        return violations

    # Local Search
    current_v = np.random.randint(0, N_group, size=15)
    current_score = evaluate(current_v)
    
    best_v = current_v.copy()
    best_score = current_score
    
    steps = 0
    max_steps = 10000
    
    while steps < max_steps and remaining() > 10:
        # Pick a violated walk
        violated_walks = []
        for w in walk_list:
            p = idx_I
            for e in w:
                p = mult_table[p, current_v[e]]
            if p == idx_I:
                violated_walks.append(w)
                
        if not violated_walks:
            break
            
        w = random.choice(violated_walks)
        # Pick an edge in the walk to change
        e = random.choice(w)
        # Try a random new value
        new_val = random.randint(0, N_group - 1)
        
        old_val = current_v[e]
        current_v[e] = new_val
        
        new_score = evaluate(current_v)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_v = current_v.copy()
            if best_score == 0:
                break
            steps = 0 # Reset steps if improvement
        else:
            # Sometimes accept worse moves (simulated annealing) or revert
            # For now, revert unless we allow hill-climbing with restarts
            current_v[e] = old_val
            steps += 1
            if steps % 100 == 0:
                # Restart
                current_v = np.random.randint(0, N_group, size=15)
                current_score = evaluate(current_v)
                steps = 0

    if best_score == 0:
        v = best_v
        n = n_base * N_group
        edges = []
        for u, v_base in pet_edges:
            e_idx = edge_index[(u, v_base)]
            vol = int(v[e_idx])
            for layer in range(N_group):
                u_lift = u * N_group + layer
                v_lift = v_base * N_group + mult_table[layer, vol]
                if u_lift < v_lift:
                    edges.append((u_lift, v_lift))
                else:
                    edges.append((v_lift, u_lift))
        
        gc = check_girth(n, edges, g, deadline - 2)
        if gc >= g:
            return (n, edges)

    # Fallback: Try a larger prime lift if PSL(2,7) fails?
    # Or just return (0, [])
    
    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
