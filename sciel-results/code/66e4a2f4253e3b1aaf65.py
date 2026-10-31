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
    
    # 3. Spectral Partitioning using Fiedler vector
    data = np.ones(len(rows), dtype=np.int32)
    row_sym = np.concatenate([rows, cols])
    col_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])
    A_sparse = csr_matrix((data_sym, (row_sym, col_sym)), shape=(N, N))
    
    D_sparse = csr_matrix(np.diag(degrees))
    L = D_sparse - A_sparse
    
    try:
        eigvals, eigvecs = eigsh(L, k=3, which='SM')
        idx = np.argsort(eigvals)
        fiedler_vec = eigvecs[:, idx[1]]
    except Exception:
        fiedler_vec = np.random.randn(N)
    
    # Get multiple spectral partitions by shifting the Fiedler vector
    def get_partition(vec, offset=0.0):
        P1 = [i for i in range(N) if vec[i] + offset > 0]
        P2 = [i for i in range(N) if vec[i] + offset <= 0]
        return P1, P2
    
    def clean_set(partition):
        # Greedy independent set with randomization to escape local traps
        shuffled = list(partition)
        random.shuffle(shuffled)
        # Sort by degree descending but with some randomness
        sorted_part = sorted(shuffled, key=lambda x: degrees[x] + random.random() * 5, reverse=True)
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
    
    def kempe_chain_local_search(S, max_iter=100):
        """Kempe-chain based local search for maximum independent set"""
        S = set(S)
        in_S = [False] * N
        for v in S:
            in_S[v] = True
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            T = [v for v in range(N) if not in_S[v]]
            if not T:
                break
            
            # Try multiple random starting points
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
                    u = queue.pop(0)
                    comp.append(u)
                    for v in adj[u]:
                        if not in_S[v] and v not in visited_comp:
                            visited_comp.add(v)
                            queue.append(v)
                            
                if len(comp) < 2:
                    continue
                    
                # Find nodes in S that conflict with comp
                R = set()
                for v in comp:
                    for u in adj[v]:
                        if in_S[u]:
                            R.add(u)
                            
                # Find maximum independent set within comp (greedy)
                sorted_comp = sorted(comp, key=lambda x: degrees[x] + random.random() * 3, reverse=True)
                I = set()
                for v in sorted_comp:
                    conflict = False
                    for u in adj[v]:
                        if u in I:
                            conflict = True
                            break
                    if not conflict:
                        I.add(v)
                        
                # Kempe flip: gain = |I| - |R|
                gain = len(I) - len(R)
                if gain > 0:
                    for v in R:
                        in_S[v] = False
                        S.discard(v)
                    for v in I:
                        in_S[v] = True
                        S.add(v)
                    improved = True
                    T = [v for v in range(N) if not in_S[v]]
                    if not T:
                        break
                        
        return S
    
    # Try multiple spectral partitions with different offsets and random perturbations
    best_set = set()
    best_size = 0
    
    # Generate multiple candidate partitions
    partitions = []
    
    # Base Fiedler partition
    P1_base, P2_base = get_partition(fiedler_vec)
    partitions.append((P1_base, "P1"))
    partitions.append((P2_base, "P2"))
    
    # Shifted partitions
    for offset in [-0.2, -0.1, 0.1, 0.2]:
        P1_shift, P2_shift = get_partition(fiedler_vec, offset)
        if len(P1_shift) != len(P1_base):
            partitions.append((P1_shift, f"P1_shift_{offset}"))
        if len(P2_shift) != len(P2_base):
            partitions.append((P2_shift, f"P2_shift_{offset}"))
    
    # Random perturbations of Fiedler vector
    for trial in range(5):
        perturbation = np.random.randn(N) * 0.1
        P1_pert, P2_pert = get_partition(fiedler_vec + perturbation)
        partitions.append((P1_pert, f"P1_pert_{trial}"))
        partitions.append((P2_pert, f"P2_pert_{trial}"))
    
    # Also try pure random partitions
    for trial in range(3):
        random_part = [i for i in range(N) if random.random() < 0.5]
        complement_part = [i for i in range(N) if i not in random_part]
        partitions.append((random_part, f"rand_{trial}"))
        partitions.append((complement_part, f"rand_comp_{trial}"))
    
    # Process each partition: clean then local search
    for partition, name in partitions:
        S_clean = clean_set(partition)
        S_final = kempe_chain_local_search(S_clean, max_iter=80)
        
        if len(S_final) > best_size:
            best_size = len(S_final)
            best_set = S_final
    
    final_vectors = []
    for i in range(N):
        if i in best_set:
            final_vectors.append(vectors[i])
            
    return np.array(final_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
