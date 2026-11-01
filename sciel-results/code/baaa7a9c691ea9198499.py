import numpy as np
import itertools
import random
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def construct():
    # 1. Generate all norm-2 integer vectors (squared norm 4)
    vectors = []
    
    # Type A: one +/- 2
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    # Type B: four +/- 1
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, val in zip(combo, signs):
                v[pos] = val
            vectors.append(v)
            
    N = len(vectors)
    
    # 2. Build conflict graph
    vec_arr = np.array(vectors, dtype=np.int8)
    dots = vec_arr @ vec_arr.T
    
    # Conflict if dot > 2
    rows, cols = np.where(dots > 2)
    valid = rows < cols
    rows = rows[valid]
    cols = cols[valid]
    
    adj = [[] for _ in range(N)]
    for r, c in zip(rows, cols):
        adj[r].append(c)
        adj[c].append(r)
        
    degrees = np.array([len(a) for a in adj], dtype=np.int32)
    
    # 3. Spectral Partitioning - compute multiple eigenvectors
    data = np.ones(len(rows), dtype=np.int32)
    row_sym = np.concatenate([rows, cols])
    col_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])
    A_sparse = csr_matrix((data_sym, (row_sym, col_sym)), shape=(N, N))
    
    D_sparse = csr_matrix(np.diag(degrees))
    L = D_sparse - A_sparse
    
    eigvecs = []
    try:
        eigvals, eigvecs_all = eigsh(L, k=6, which='SM')
        idx = np.argsort(eigvals)
        # Skip the trivial eigenvector (index 0), use indices 1-5
        for i in idx[1:6]:
            eigvecs.append(eigvecs_all[:, i])
    except Exception:
        eigvecs = [np.random.randn(N) for _ in range(5)]
    
    def clean_independent_set(partition):
        """Greedy independent set from a candidate partition, using degree-based ordering."""
        # Use a heap or sort by degree
        # Sorting is fine for N~5000
        sorted_part = sorted(partition, key=lambda x: degrees[x], reverse=True)
        s = set()
        in_s = [False] * N
        for v in sorted_part:
            conflict = False
            for u in adj[v]:
                if in_s[u]:
                    conflict = True
                    break
            if not conflict:
                s.add(v)
                in_s[v] = True
        return s

    def randomized_kempe_local_search(S, max_iter=500):
        """
        Randomized Kempe-chain based local search.
        1. Try to add any vertex from T that has no neighbors in S.
        2. If none, pick a random vertex from T, find its component in G[T],
           evaluate swap (remove neighbors in S, add IS of component).
        """
        S = set(S)
        in_S = [False] * N
        for v in S:
            in_S[v] = True
        
        T = [v for v in range(N) if not in_S[v]]
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Step 1: Fast additions
            # Check if any vertex in T has no neighbors in S
            # We can maintain a list of "available" vertices, but T changes.
            # For efficiency, just scan T occasionally or maintain a count.
            # Given N is small, scanning is okay, but let's be smart.
            
            added_any = False
            new_T = []
            for v in T:
                # Check if v can be added
                can_add = True
                for u in adj[v]:
                    if in_S[u]:
                        can_add = False
                        break
                if can_add:
                    in_S[v] = True
                    S.add(v)
                    improved = True
                    added_any = True
                else:
                    new_T.append(v)
            
            if added_any:
                T = new_T
                continue
            
            # Step 2: Kempe Chain Swap
            if not T:
                break
            
            # Pick a random vertex from T to explore its component
            # To avoid re-evaluating the same failed component repeatedly, 
            # we could keep a 'blocked' set, but for simplicity, random sampling works well.
            
            # We will try up to 10 random vertices in this iteration
            found_swap = False
            for _ in range(10):
                if not T:
                    break
                start_v = random.choice(T)
                
                # BFS to find component C in G[T]
                C = []
                visited = {start_v}
                queue = [start_v]
                while queue:
                    u = queue.pop(0)
                    C.append(u)
                    for v in adj[u]:
                        if not in_S[v] and v not in visited:
                            visited.add(v)
                            queue.append(v)
                
                # Calculate R: neighbors of C in S
                R = set()
                for v in C:
                    for u in adj[v]:
                        if in_S[u]:
                            R.add(u)
                
                # Calculate I: Max IS in C (Greedy)
                # Sort C by degree (in original graph or induced? Original is fine)
                sorted_C = sorted(C, key=lambda x: degrees[x], reverse=True)
                I = set()
                in_I = [False] * N
                for v in sorted_C:
                    conflict = False
                    for u in adj[v]:
                        if in_I[u]:
                            conflict = True
                            break
                    if not conflict:
                        I.add(v)
                        in_I[v] = True
                
                gain = len(I) - len(R)
                if gain > 0:
                    # Perform swap
                    for v in R:
                        in_S[v] = False
                        S.discard(v)
                    for v in I:
                        in_S[v] = True
                        S.add(v)
                    
                    # Update T
                    # R vertices are now in T
                    # I vertices are now in S (remove from T)
                    # C \ I are still in T
                    new_T = []
                    for v in T:
                        if v in I:
                            continue
                        new_T.append(v)
                    for v in R:
                        new_T.append(v)
                    T = new_T
                    
                    improved = True
                    found_swap = True
                    break
            
            if not found_swap and not improved:
                # If we tried 10 random vertices and found no swap, 
                # we might be stuck. 
                # To escape, we can do a random move (accept worse solution) 
                # or just break if we've been stuck for a while.
                # For now, let's just break to avoid infinite loops on stuck states.
                # A more robust approach would be simulated annealing.
                break
                
        return S

    def spectral_partition(evec, num_parts=2):
        """
        Partition vertices based on eigenvector values.
        """
        evec = evec - np.mean(evec)
        if np.std(evec) == 0:
            return [[i for i in range(N) if i % 2 == 0], [i for i in range(N) if i % 2 == 1]]
        
        if num_parts == 2:
            P1 = [i for i in range(N) if evec[i] > 0]
            P2 = [i for i in range(N) if evec[i] <= 0]
            return [P1, P2]
        else:
            thresholds = np.percentile(evec, np.linspace(0, 100, num_parts + 1))
            parts = []
            for k in range(num_parts):
                part = [i for i in range(N) if thresholds[k] <= evec[i] < thresholds[k+1]]
                parts.append(part)
            return parts

    best_set = set()
    best_size = 0
    
    # Strategy: Multiple restarts with different spectral partitions
    num_restarts = 10
    for r in range(num_restarts):
        # Select an eigenvector or combination
        if r < len(eigvecs):
            evec = eigvecs[r]
        else:
            # Mix of eigenvectors
            evec = sum(eigvecs[i] for i in range(min(3, len(eigvecs)))) / 3.0
        
        # Add noise to escape symmetry
        noise = np.random.randn(N) * 0.01
        evec = evec + noise
        
        parts = spectral_partition(evec, 2)
        
        # Clean both parts
        S1 = clean_independent_set(parts[0])
        S2 = clean_independent_set(parts[1])
        
        # Start with the larger one
        current = S1 if len(S1) >= len(S2) else S2
        
        # Merge the other into current
        # Try to add vertices from the other set that don't conflict
        other = S2 if len(S1) >= len(S2) else S1
        in_current = [False] * N
        for v in current:
            in_current[v] = True
            
        for v in other:
            if in_current[v]:
                continue
            conflict = False
            for u in adj[v]:
                if in_current[u]:
                    conflict = True
                    break
            if not conflict:
                in_current[v] = True
                current.add(v)
        
        # Run Local Search
        current = randomized_kempe_local_search(current)
        
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
            
    # Final repair: try to add any remaining vertices
    in_best = [False] * N
    for v in best_set:
        in_best[v] = True
    
    for v in range(N):
        if in_best[v]:
            continue
        can_add = True
        for u in adj[v]:
            if in_best[u]:
                can_add = False
                break
        if can_add:
            in_best[v] = True
            best_set.add(v)
    
    final_vectors = []
    for i in range(N):
        if i in best_set:
            final_vectors.append(vectors[i])
            
    return np.array(final_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
