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
    
    # 3. Spectral Partitioning - compute more eigenvectors for better embedding
    data = np.ones(len(rows), dtype=np.int32)
    row_sym = np.concatenate([rows, cols])
    col_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])
    A_sparse = csr_matrix((data_sym, (row_sym, col_sym)), shape=(N, N))
    
    D_sparse = csr_matrix(np.diag(degrees))
    L = D_sparse - A_sparse
    
    eigvecs = []
    try:
        eigvals, eigvecs_all = eigsh(L, k=8, which='SM')
        idx = np.argsort(eigvals)
        # Skip the trivial eigenvector (index 0), use indices 1-7
        for i in idx[1:8]:
            eigvecs.append(eigvecs_all[:, i])
    except Exception:
        eigvecs = [np.random.randn(N) for _ in range(7)]
    
    def clean_independent_set(partition):
        """Greedy independent set from a candidate partition, using degree-based ordering."""
        sorted_part = sorted(partition, key=lambda x: degrees[x], reverse=True)
        s = set()
        for v in sorted_part:
            conflict = False
            for u in adj[v]:
                if u in s:
                    conflict = True
                    break
            if not conflict:
                s.add(v)
        return s
    
    def greedy_independent_set_from_candidates(candidates):
        """Greedy IS from a list of candidate vertices."""
        sorted_cand = sorted(candidates, key=lambda x: degrees[x], reverse=True)
        s = set()
        for v in sorted_cand:
            conflict = False
            for u in adj[v]:
                if u in s:
                    conflict = True
                    break
            if not conflict:
                s.add(v)
        return s
    
    def kempe_chain_local_search(S, max_iter=300):
        """
        Enhanced Kempe-chain based local search to maximize independent set size.
        Uses multi-component evaluation and tries individual vertex swaps too.
        """
        S = set(S)
        in_S = [False] * N
        for v in S:
            in_S[v] = True
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Collect all vertices not in S
            T = [v for v in range(N) if not in_S[v]]
            if not T:
                break
            
            # Find connected components in the induced subgraph on T
            visited_comp = set()
            components = []
            
            for start_v in T:
                if start_v in visited_comp:
                    continue
                
                comp = []
                queue = [start_v]
                visited_comp.add(start_v)
                
                while queue:
                    u = queue.pop(0)
                    comp.append(u)
                    for v in adj[u]:
                        if not in_S[v] and v not in visited_comp:
                            visited_comp.add(v)
                            queue.append(v)
                
                components.append(comp)
            
            # Try individual vertex additions first (fast wins)
            for v in T:
                if in_S[v]:
                    continue
                # Check if v can be added (no conflicts with S)
                can_add = True
                for u in adj[v]:
                    if in_S[u]:
                        can_add = False
                        break
                if can_add:
                    in_S[v] = True
                    S.add(v)
                    improved = True
            
            if improved:
                continue
            
            # For each component, compute the gain of swapping
            best_gain = 0
            best_comp = None
            best_I = None
            best_R = None
            
            for comp in components:
                # Compute neighbors in S
                R = set()
                for v in comp:
                    for u in adj[v]:
                        if in_S[u]:
                            R.add(u)
                
                # Compute max independent set within comp (greedy)
                sorted_comp = sorted(comp, key=lambda x: degrees[x], reverse=True)
                I = set()
                for v in sorted_comp:
                    conflict = False
                    for u in adj[v]:
                        if u in I:
                            conflict = True
                            break
                    if not conflict:
                        I.add(v)
                
                gain = len(I) - len(R)
                if gain > best_gain:
                    best_gain = gain
                    best_comp = comp
                    best_I = I
                    best_R = R
            
            # Also try single-vertex swaps: remove one from S, add one from T
            if best_gain <= 0:
                # Try removing low-degree vertices from S and adding high-degree from T
                S_sorted = sorted(S, key=lambda x: degrees[x])
                T_sorted = sorted(T, key=lambda x: degrees[x], reverse=True)
                
                for v_remove in S_sorted[:15]:  # Try removing lowest degree first
                    for v_add in T_sorted[:15]:  # Try adding highest degree
                        # Check if v_add conflicts with S \ {v_remove}
                        can_add = True
                        for u in adj[v_add]:
                            if in_S[u] and u != v_remove:
                                can_add = False
                                break
                        if can_add:
                            # Perform swap
                            in_S[v_remove] = False
                            S.discard(v_remove)
                            in_S[v_add] = True
                            S.add(v_add)
                            improved = True
                            break
                    if improved:
                        break
            
            # Try 2-vertex swaps: remove two from S, add two from T
            if not improved:
                S_sorted2 = sorted(S, key=lambda x: degrees[x])
                T_sorted2 = sorted(T, key=lambda x: degrees[x], reverse=True)
                
                found_swap = False
                for i in range(min(10, len(S_sorted2))):
                    for j in range(i+1, min(10, len(S_sorted2))):
                        v1, v2 = S_sorted2[i], S_sorted2[j]
                        for k in range(min(10, len(T_sorted2))):
                            for l in range(k+1, min(10, len(T_sorted2))):
                                w1, w2 = T_sorted2[k], T_sorted2[l]
                                # Check if w1, w2 can be added after removing v1, v2
                                can_add1 = True
                                can_add2 = True
                                for u in adj[w1]:
                                    if in_S[u] and u not in (v1, v2):
                                        can_add1 = False
                                        break
                                for u in adj[w2]:
                                    if in_S[u] and u not in (v1, v2):
                                        can_add2 = False
                                        break
                                if can_add1 and can_add2:
                                    # Check w1 and w2 don't conflict with each other
                                    if w2 not in adj[w1]:
                                        # Perform 2-swap
                                        in_S[v1] = False
                                        in_S[v2] = False
                                        S.discard(v1)
                                        S.discard(v2)
                                        in_S[w1] = True
                                        in_S[w2] = True
                                        S.add(w1)
                                        S.add(w2)
                                        improved = True
                                        found_swap = True
                                        break
                            if found_swap:
                                break
                    if found_swap:
                        break
            
            if not improved and best_comp is not None and best_gain > 0:
                # Perform the component swap
                for v in best_R:
                    in_S[v] = False
                    S.discard(v)
                for v in best_I:
                    in_S[v] = True
                    S.add(v)
                improved = True
        
        return S
    
    def merge_two_sets(S1, S2):
        """
        Merge two independent sets by starting with the larger one
        and trying to add vertices from the other that don't conflict.
        """
        S = set(S1) if len(S1) >= len(S2) else set(S2)
        other = set(S2) if len(S1) >= len(S2) else set(S1)
        
        in_S = [False] * N
        for v in S:
            in_S[v] = True
        
        # Try to add vertices from other that don't conflict
        for v in other:
            if in_S[v]:
                continue
            conflict = False
            for u in adj[v]:
                if in_S[u]:
                    conflict = True
                    break
            if not conflict:
                in_S[v] = True
                S.add(v)
        
        # Run Kempe chain local search
        S = kempe_chain_local_search(S)
        return S
    
    def spectral_partition(evec, num_parts=2):
        """
        Partition vertices based on eigenvector values.
        Returns list of partitions.
        """
        # Normalize eigenvector
        evec = evec - np.mean(evec)
        if np.std(evec) == 0:
            return [[i for i in range(N) if i % 2 == 0], [i for i in range(N) if i % 2 == 1]]
        
        # For 2 parts: sign split
        if num_parts == 2:
            P1 = [i for i in range(N) if evec[i] > 0]
            P2 = [i for i in range(N) if evec[i] <= 0]
            return [P1, P2]
        else:
            # For more parts: use quantiles
            thresholds = np.percentile(evec, np.linspace(0, 100, num_parts + 1))
            parts = []
            for k in range(num_parts):
                part = [i for i in range(N) if thresholds[k] <= evec[i] < thresholds[k+1]]
                parts.append(part)
            return parts
    
    def spectral_embedding_partition(eigvecs_list, num_parts=2):
        """
        Use multiple eigenvectors for a more robust partition.
        Combines eigenvector values via weighted sum.
        """
        if not eigvecs_list:
            return None
        
        # Weighted combination of eigenvectors
        weights = np.ones(len(eigvecs_list))
        combined = np.zeros(N)
        for i, evec in enumerate(eigvecs_list):
            evec_norm = evec - np.mean(evec)
            combined += weights[i] * evec_norm
        
        return spectral_partition(combined, num_parts)
    
    # 4. Generate multiple partitions and find the best
    best_set = set()
    best_size = 0
    
    all_partitions = []
    
    # Strategy 1: Use each eigenvector for 2-way partition
    for evec in eigvecs:
        parts = spectral_partition(evec, 2)
        if len(parts) == 2:
            all_partitions.append(parts)
    
    # Strategy 2: Use combinations of eigenvectors for 4-way partition
    for i in range(len(eigvecs)):
        for j in range(i+1, len(eigvecs)):
            # Create 2D embedding
            X = np.column_stack([eigvecs[i], eigvecs[j]])
            X = X - X.mean(axis=0)
            
            # K-means style: split by quadrants
            medians = np.median(X, axis=0)
            parts = [[], [], [], []]
            for idx in range(N):
                if X[idx, 0] >= medians[0] and X[idx, 1] >= medians[1]:
                    parts[0].append(idx)
                elif X[idx, 0] < medians[0] and X[idx, 1] >= medians[1]:
                    parts[1].append(idx)
                elif X[idx, 0] >= medians[0] and X[idx, 1] < medians[1]:
                    parts[2].append(idx)
                else:
                    parts[3].append(idx)
            
            # Merge pairs to get 2-way partitions
            all_partitions.append([parts[0] + parts[1], parts[2] + parts[3]])
            all_partitions.append([parts[0] + parts[2], parts[1] + parts[3]])
            all_partitions.append([parts[0] + parts[3], parts[1] + parts[2]])
    
    # Strategy 3: Random perturbed eigenvectors
    for evec in eigvecs:
        for restart in range(5):
            noise = np.random.randn(N) * 0.05
            perturbed = evec + noise
            parts = spectral_partition(perturbed, 2)
            if len(parts) == 2:
                all_partitions.append(parts)
    
    # Strategy 4: Multi-eigenvector combined partitions
    for i in range(len(eigvecs)):
        for j in range(i+1, len(eigvecs)):
            combined_parts = spectral_embedding_partition([eigvecs[i], eigvecs[j]], 2)
            if combined_parts and len(combined_parts) == 2:
                all_partitions.append(combined_parts)
    
    # Strategy 5: Use top 3 eigenvectors combined
    if len(eigvecs) >= 3:
        for combo in itertools.combinations(range(len(eigvecs)), 3):
            combined_parts = spectral_embedding_partition([eigvecs[i] for i in combo], 2)
            if combined_parts and len(combined_parts) == 2:
                all_partitions.append(combined_parts)
    
    # Process all partitions
    for parts in all_partitions:
        if len(parts) < 2:
            continue
        
        # Clean each part
        cleaned_parts = []
        for part in parts:
            if not part:
                cleaned_parts.append(set())
            else:
                cleaned_parts.append(clean_independent_set(part))
        
        # Merge all cleaned parts
        if not cleaned_parts or all(len(p) == 0 for p in cleaned_parts):
            continue
        
        # Start with the largest
        current = max(cleaned_parts, key=len)
        for other in cleaned_parts:
            if other is current:
                continue
            current = merge_two_sets(current, other)
        
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # Additional: Try direct greedy on full vertex set with spectral ordering
    # Order vertices by Fiedler vector, then greedy
    fiedler_vec = eigvecs[0]
    order = np.argsort(fiedler_vec)
    greedy_set = greedy_independent_set_from_candidates(list(order))
    greedy_set = kempe_chain_local_search(greedy_set)
    
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(greedy_set)
    
    # Additional: Try reverse spectral ordering
    order_rev = np.argsort(-fiedler_vec)
    greedy_set_rev = greedy_independent_set_from_candidates(list(order_rev))
    greedy_set_rev = kempe_chain_local_search(greedy_set_rev)
    
    if len(greedy_set_rev) > best_size:
        best_set = greedy_set_rev
        best_size = len(greedy_set_rev)
    
    # Additional: Try multiple random orderings combined with spectral
    for restart in range(3):
        random_order = list(range(N))
        random.shuffle(random_order)
        # Mix spectral and random
        mixed_order = []
        for i in range(N):
            if i % 2 == 0:
                mixed_order.append(order[i])
            else:
                mixed_order.append(random_order[i])
        greedy_set_mixed = greedy_independent_set_from_candidates(mixed_order)
        greedy_set_mixed = kempe_chain_local_search(greedy_set_mixed)
        
        if len(greedy_set_mixed) > best_size:
            best_set = greedy_set_mixed
            best_size = len(greedy_set_mixed)
    
    # Final repair: try to add any remaining vertices (multiple passes)
    in_best = [False] * N
    for v in best_set:
        in_best[v] = True
    
    for _ in range(3):  # Multiple repair passes
        any_added = False
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
                any_added = True
        if not any_added:
            break
    
    # Final Kempe chain polish
    best_set = kempe_chain_local_search(best_set, max_iter=100)
    
    final_vectors = []
    for i in range(N):
        if i in best_set:
            final_vectors.append(vectors[i])
            
    return np.array(final_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
