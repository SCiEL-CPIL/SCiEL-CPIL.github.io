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
    
    # 3. Spectral Partitioning with multiple eigenvectors
    data = np.ones(len(rows), dtype=np.int32)
    row_sym = np.concatenate([rows, cols])
    col_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])
    A_sparse = csr_matrix((data_sym, (row_sym, col_sym)), shape=(N, N))
    
    D_sparse = csr_matrix(np.diag(degrees))
    L = D_sparse - A_sparse
    
    # Get multiple eigenvectors for better partitioning
    try:
        eigvals, eigvecs = eigsh(L, k=4, which='SM')
        idx = np.argsort(eigvals)
        # Use Fiedler vector (second smallest eigenvalue)
        fiedler_vec = eigvecs[:, idx[1]]
        # Also get second Fiedler vector for refinement
        fiedler_vec2 = eigvecs[:, idx[2]]
    except Exception:
        fiedler_vec = np.random.randn(N)
        fiedler_vec2 = np.random.randn(N)
        
    def clean_set(partition):
        """Greedy independent set using degree ordering with tie-breaking by connectivity"""
        # Sort by degree descending, break ties by number of conflicts with already selected
        s = set()
        in_s = [False] * N
        # Precompute conflict counts for tie-breaking
        conflict_count = [0] * N
        
        # Process in multiple passes with different orderings
        best_result = set()
        
        # Pass 1: degree ordering
        sorted_part = sorted(partition, key=lambda x: (-degrees[x], x))
        s1 = set()
        in_s1 = [False] * N
        for v in sorted_part:
            conflict = False
            for u in adj[v]:
                if in_s1[u]:
                    conflict = True
                    break
            if not conflict:
                s1.add(v)
                in_s1[v] = True
        best_result = s1
        
        # Pass 2: reverse degree ordering (sometimes helps)
        sorted_part2 = sorted(partition, key=lambda x: (degrees[x], x))
        s2 = set()
        in_s2 = [False] * N
        for v in sorted_part2:
            conflict = False
            for u in adj[v]:
                if in_s2[u]:
                    conflict = True
                    break
            if not conflict:
                s2.add(v)
                in_s2[v] = True
        if len(s2) > len(best_result):
            best_result = s2
            
        return best_result
        
    def local_search(S, max_iter=100):
        """Kempe-chain based local search for maximum independent set"""
        S = set(S)
        in_S = [False] * N
        for v in S:
            in_S[v] = True
            
        improved = True
        iter_count = 0
        no_improve_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            no_improve_count += 1
            
            T = [v for v in range(N) if not in_S[v]]
            if not T:
                break
            
            # Try multiple starting points
            samples = min(20, len(T))
            for _ in range(samples):
                if not T:
                    break
                start_v = random.choice(T)
                
                # Find connected component in complement graph
                comp = []
                queue = [start_v]
                visited_comp = {start_v}
                
                while queue:
                    u = queue.pop()
                    comp.append(u)
                    for v in adj[u]:
                        if not in_S[v] and v not in visited_comp:
                            visited_comp.add(v)
                            queue.append(v)
                            
                if len(comp) < 2:
                    continue
                    
                # Find nodes in S that conflict with this component
                R = set()
                for v in comp:
                    for u in adj[v]:
                        if in_S[u]:
                            R.add(u)
                            
                # Find maximum independent set within comp using greedy
                sorted_comp = sorted(comp, key=lambda x: (-degrees[x], x))
                I = set()
                in_I = [False] * N
                for v in sorted_comp:
                    conflict = False
                    for u in adj[v]:
                        if in_I[u]:
                            conflict = True
                            break
                    if not conflict:
                        I.add(v)
                        in_I[v] = True
                        
                # Perform Kempe flip if beneficial
                if not R:
                    if I:
                        for v in I:
                            in_S[v] = True
                            S.add(v)
                        improved = True
                        no_improve_count = 0
                else:
                    # Compare sizes with a threshold
                    if len(I) > len(R) + 1:  # Require clear improvement
                        for v in R:
                            in_S[v] = False
                            S.remove(v)
                        for v in I:
                            in_S[v] = True
                            S.add(v)
                        improved = True
                        no_improve_count = 0
                        
            if not improved:
                if no_improve_count >= 5:
                    break
                no_improve_count += 1
        return S
        
    def multi_start_local_search(initial_set, n_starts=3):
        """Run local search from multiple perturbations of the initial set"""
        best = local_search(set(initial_set))
        best_size = len(best)
        
        for start in range(n_starts - 1):
            # Perturb the initial set
            perturbed = set(initial_set)
            # Remove some random elements
            remove_count = min(len(perturbed) // 10, 20)
            to_remove = random.sample(list(perturbed), min(remove_count, len(perturbed)))
            for v in to_remove:
                perturbed.discard(v)
            # Add some random non-conflicting elements
            candidates = [v for v in range(N) if v not in perturbed]
            if candidates:
                add_count = min(len(candidates) // 10, 20)
                to_add = random.sample(candidates, min(add_count, len(candidates)))
                for v in to_add:
                    # Check if v conflicts with any element in perturbed
                    conflict = False
                    for u in adj[v]:
                        if u in perturbed:
                            conflict = True
                            break
                    if not conflict:
                        perturbed.add(v)
                        
            result = local_search(perturbed)
            if len(result) > best_size:
                best = result
                best_size = len(result)
                
        return best
        
    best_set = set()
    best_size = 0
    
    # Generate multiple spectral partitions
    partitions = []
    
    # Primary Fiedler partition
    P1 = [i for i in range(N) if fiedler_vec[i] > 0]
    P2 = [i for i in range(N) if fiedler_vec[i] <= 0]
    partitions.append(P1)
    partitions.append(P2)
    
    # Secondary Fiedler partition
    P3 = [i for i in range(N) if fiedler_vec2[i] > 0]
    P4 = [i for i in range(N) if fiedler_vec2[i] <= 0]
    partitions.append(P3)
    partitions.append(P4)
    
    # Mixed partitions using both eigenvectors
    P5 = [i for i in range(N) if fiedler_vec[i] + fiedler_vec2[i] > 0]
    P6 = [i for i in range(N) if fiedler_vec[i] + fiedler_vec2[i] <= 0]
    partitions.append(P5)
    partitions.append(P6)
    
    P7 = [i for i in range(N) if fiedler_vec[i] - fiedler_vec2[i] > 0]
    P8 = [i for i in range(N) if fiedler_vec[i] - fiedler_vec2[i] <= 0]
    partitions.append(P7)
    partitions.append(P8)
    
    # Try each partition with multi-start local search
    for partition in partitions:
        if not partition:
            continue
        S_clean = clean_set(partition)
        S_final = multi_start_local_search(S_clean, n_starts=2)
        if len(S_final) > best_size:
            best_set = S_final
            best_size = len(S_final)
            
    # Also try a global greedy approach as baseline
    all_nodes = list(range(N))
    sorted_all = sorted(all_nodes, key=lambda x: (-degrees[x], x))
    global_greedy = set()
    in_global = [False] * N
    for v in sorted_all:
        conflict = False
        for u in adj[v]:
            if in_global[u]:
                conflict = True
                break
        if not conflict:
            global_greedy.add(v)
            in_global[v] = True
    
    if len(global_greedy) > best_size:
        best_set = global_greedy
        best_size = len(best_set)
        
    # Final local search on the best set
    best_set = multi_start_local_search(best_set, n_starts=2)
    
    final_vectors = []
    for i in range(N):
        if i in best_set:
            final_vectors.append(vectors[i])
            
    return np.array(final_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
