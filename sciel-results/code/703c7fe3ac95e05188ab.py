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
    D = 11
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
    
    # 3. Spectral Partitioning - compute Fiedler vector
    data = np.ones(len(rows), dtype=np.int32)
    row_sym = np.concatenate([rows, cols])
    col_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])
    A_sparse = csr_matrix((data_sym, (row_sym, col_sym)), shape=(N, N))
    
    D_sparse = csr_matrix(np.diag(degrees))
    L = D_sparse - A_sparse
    
    eigvecs = []
    try:
        # Get smallest non-zero eigenvalues
        eigvals, eigvecs_all = eigsh(L, k=5, which='SM')
        idx = np.argsort(eigvals)
        # Skip the trivial eigenvector (index 0)
        for i in idx[1:5]:
            eigvecs.append(eigvecs_all[:, i])
    except Exception:
        eigvecs = [np.random.randn(N) for _ in range(4)]
    
    def greedy_independent_set(order):
        """Greedy IS given an order of vertices."""
        s = set()
        in_s = [False] * N
        for v in order:
            if in_s[v]:
                continue
            conflict = False
            for u in adj[v]:
                if in_s[u]:
                    conflict = True
                    break
            if not conflict:
                s.add(v)
                in_s[v] = True
        return s

    def kempe_chain_local_search(S, max_iter=50):
        """
        Kempe-chain based local search to maximize independent set size.
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
            
            # 1. Try to add any vertex that doesn't conflict
            for v in range(N):
                if in_S[v]:
                    continue
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
            
            # 2. Find connected components in V \ S
            T = [v for v in range(N) if not in_S[v]]
            if not T:
                break
                
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
            
            # 3. For each component, evaluate swap gain
            best_gain = 0
            best_I = None
            best_R = None
            
            for comp in components:
                # Neighbors in S
                R = set()
                for v in comp:
                    for u in adj[v]:
                        if in_S[u]:
                            R.add(u)
                
                # Max IS in comp (greedy by degree)
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
                    best_I = I
                    best_R = R
            
            # 4. Execute best swap
            if best_gain > 0 and best_I is not None:
                for v in best_R:
                    in_S[v] = False
                    S.discard(v)
                for v in best_I:
                    in_S[v] = True
                    S.add(v)
                improved = True
            
            # 5. If no gain, try single vertex swaps to escape local minima
            if not improved:
                S_list = list(S)
                T_list = T
                if S_list and T_list:
                    S_sorted = sorted(S_list, key=lambda x: degrees[x])
                    T_sorted = sorted(T_list, key=lambda x: degrees[x], reverse=True)
                    
                    found_swap = False
                    for u in S_sorted[:5]:
                        for v in T_sorted[:10]:
                            if in_S[v]: continue
                            can_add = True
                            for w in adj[v]:
                                if in_S[w] and w != u:
                                    can_add = False
                                    break
                            if can_add:
                                in_S[u] = False
                                S.discard(u)
                                in_S[v] = True
                                S.add(v)
                                improved = True
                                found_swap = True
                                break
                        if found_swap:
                            break
        
        return S

    # 4. Multi-start optimization
    best_set = set()
    best_size = 0
    
    orderings = []
    
    for evec in eigvecs:
        order_asc = list(np.argsort(evec))
        order_desc = list(np.argsort(-evec))
        orderings.append(order_asc)
        orderings.append(order_desc)
        
    order_deg = list(np.argsort(degrees))
    orderings.append(order_deg)
    orderings.append(list(np.argsort(-degrees)))
    
    for _ in range(5):
        random_order = list(range(N))
        random.shuffle(random_order)
        orderings.append(random_order)
    
    for order in orderings:
        S = greedy_independent_set(order)
        S = kempe_chain_local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
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
