"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import time

def construct():
    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def find_sum_free_spectral(elements):
        """
        Find a large sum-free subset using spectral max-cut on the Schur graph.
        """
        n = len(elements)
        if n == 0:
            return []
        
        elem_set = set(elements)
        elem_to_idx = {v: i for i, v in enumerate(elements)}
        
        # Build edge list for max-cut
        # For each triple (a,b,c) with a+b=c, add edges to encourage separation
        edge_dict = {}  # (i,j) -> weight, with i < j
        
        for i in range(n):
            for j in range(i+1, n):
                a, b = elements[i], elements[j]
                s = a + b
                if s in elem_to_idx:
                    k = elem_to_idx[s]
                    # Add edges (i,j), (i,k), (j,k)
                    pairs = [(min(i,j), max(i,j)), (min(i,k), max(i,k)), (min(j,k), max(j,k))]
                    for p in pairs:
                        edge_dict[p] = edge_dict.get(p, 0) + 1
        
        # Handle a+a=c
        for i in range(n):
            a = elements[i]
            s = 2 * a
            if s in elem_to_idx:
                k = elem_to_idx[s]
                if i != k:
                    p = (min(i,k), max(i,k))
                    edge_dict[p] = edge_dict.get(p, 0) + 2
        
        if not edge_dict:
            return list(elements)
        
        # Build adjacency structure for fast flip computation
        adj = [[] for _ in range(n)]  # adj[i] = list of (j, weight)
        for (i, j), w in edge_dict.items():
            adj[i].append((j, w))
            adj[j].append((i, w))
        
        # Build sparse matrix for spectral method
        A = lil_matrix((n, n), dtype=np.float64)
        for (i, j), w in edge_dict.items():
            A[i, j] = w
            A[j, i] = w
        A = A.tocsr()
        
        # Spectral initialization
        try:
            from scipy.sparse.linalg import eigsh
            if n > 2:
                k_eig = min(n, 10)
                eigenvalues, eigenvectors = eigsh(A, k=k_eig, which='SM')
                idx = np.argsort(eigenvalues)
                fiedler_vec = eigenvectors[:, idx[1]]
            else:
                fiedler_vec = np.array([1.0, -1.0])
        except:
            import random
            rng = random.Random(42)
            fiedler_vec = np.array([1.0 if rng.random() < 0.5 else -1.0 for _ in range(n)])
        
        colors = (fiedler_vec > 0).astype(int)
        
        def local_search(colors):
            """Local search to maximize cut."""
            improved = True
            max_iter = 100
            for _ in range(max_iter):
                if not improved:
                    break
                improved = False
                for i in range(n):
                    # Compute gain from flipping i
                    gain = 0.0
                    for j, w in adj[i]:
                        if colors[i] != colors[j]:
                            gain -= w
                        else:
                            gain += w
                    if gain > 0:
                        colors[i] = 1 - colors[i]
                        improved = True
            return colors
        
        def make_sum_free(subset):
            """Remove elements to make subset sum-free."""
            s_set = set(subset)
            result = list(subset)
            changed = True
            while changed:
                changed = False
                for i in range(len(result)):
                    for j in range(i, len(result)):
                        s = result[i] + result[j]
                        if s in s_set:
                            result.remove(s)
                            s_set.discard(s)
                            changed = True
                            break
                    if changed:
                        break
            return result
        
        def try_add_elements(base_subset, candidates):
            """Try to add elements from candidates to base_subset while maintaining sum-free."""
            current = set(base_subset)
            candidates = list(candidates)
            added = True
            while added:
                added = False
                for x in candidates[:]:
                    conflict = False
                    for a in current:
                        if a + x in current:
                            conflict = True
                            break
                        if 2 * a == x:
                            conflict = True
                            break
                    if not conflict:
                        for a in current:
                            for b in current:
                                if a + b == x:
                                    conflict = True
                                    break
                            if conflict:
                                break
                    if not conflict:
                        current.add(x)
                        candidates.remove(x)
                        added = True
            return list(current)
        
        best_size = 0
        best_subset = []
        
        # Process initial coloring
        colors = local_search(colors)
        class0 = [elements[i] for i in range(n) if colors[i] == 0]
        class1 = [elements[i] for i in range(n) if colors[i] == 1]
        
        for cls in [class0, class1]:
            sf = make_sum_free(cls)
            if len(sf) > best_size:
                best_size = len(sf)
                best_subset = sf
        
        if len(class0) >= len(class1):
            base, cand = class0, class1
        else:
            base, cand = class1, class0
        combined = try_add_elements(base, cand)
        if len(combined) > best_size:
            best_size = len(combined)
            best_subset = combined
        
        # Random restarts
        import random
        rng_local = random.Random(123)
        
        for restart in range(15):
            colors = np.array([1 if rng_local.random() < 0.5 else 0 for _ in range(n)])
            colors = local_search(colors)
            
            class0 = [elements[i] for i in range(n) if colors[i] == 0]
            class1 = [elements[i] for i in range(n) if colors[i] == 1]
            
            for cls in [class0, class1]:
                sf = make_sum_free(cls)
                if len(sf) > best_size:
                    best_size = len(sf)
                    best_subset = sf
            
            if len(class0) >= len(class1):
                base, cand = class0, class1
            else:
                base, cand = class1, class0
            combined = try_add_elements(base, cand)
            if len(combined) > best_size:
                best_size = len(combined)
                best_subset = combined
        
        return best_subset

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elements = instance(size, seed)
        subset = find_sum_free_spectral(elements)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
