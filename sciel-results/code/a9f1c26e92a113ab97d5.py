import time
import random
import itertools

def construct():
    deadline = time.time() + 140  # Safety margin
    
    # Pre-compute masks for subsets of size 6 to n
    # We will generate them on the fly per n to save memory if n changes, 
    # but since we try 21 then 20, we can just generate for the current n.
    
    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m -= lsb
        return verts

    def is_regular_subset(adj, mask):
        verts = get_vertices(mask)
        if len(verts) < 6:
            return False
        
        degs = []
        for v in verts:
            degs.append((adj[v] & mask).bit_count())
        
        first = degs[0]
        for d in degs[1:]:
            if d != first:
                return False
        return True

    def local_search(n, max_restarts=10):
        """Perform local search for a valid graph on n vertices."""
        # Pre-compute masks for each size k >= 6
        masks_by_size = {}
        for k in range(6, n + 1):
            masks = []
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                masks.append(mask)
            masks_by_size[k] = masks

        for _ in range(max_restarts):
            if time.time() >= deadline:
                return None
                
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search loop
            iterations = 0
            max_iterations = 100
            while iterations < max_iterations and time.time() < deadline:
                bad_mask = None
                # Check sizes in order
                for k in range(6, n + 1):
                    for mask in masks_by_size[k]:
                        if is_regular_subset(adj, mask):
                            bad_mask = mask
                            break
                    if bad_mask:
                        break
                
                if bad_mask is None:
                    return adj
                
                # Pick a random edge within the bad subset to flip
                verts = get_vertices(bad_mask)
                u, v = random.sample(verts, 2)
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                iterations += 1
        return None

    # Try to find a graph with n=21
    adj = local_search(21)
    if adj is not None:
        edges = []
        for u in range(21):
            for v in range(u + 1, 21):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return {'n': 21, 'edges': edges}

    # Try to find a graph with n=20
    adj = local_search(20)
    if adj is not None:
        edges = []
        for u in range(20):
            for v in range(u + 1, 20):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return {'n': 20, 'edges': edges}

    # Fallback
    return {'n': 5, 'edges': []}
