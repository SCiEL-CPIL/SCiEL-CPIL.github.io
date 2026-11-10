import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    
    def common_count(mask, i, j, k):
        return (mask[i] & mask[j] & mask[k]).bit_count()
    
    def is_valid(mask):
        for i in range(n):
            mi = mask[i]
            for j in range(i + 1, n):
                mij = mi & mask[j]
                for k in range(j + 1, n):
                    if (mij & mask[k]).bit_count() >= 3:
                        return False
        return True
    
    def edge_count(mask):
        return sum(mask[i].bit_count() for i in range(n)) // 2
    
    def find_violations(mask):
        viols = []
        for i in range(n):
            mi = mask[i]
            for j in range(i + 1, n):
                mij = mi & mask[j]
                for k in range(j + 1, n):
                    if (mij & mask[k]).bit_count() >= 3:
                        viols.append((i, j, k))
        return viols

    def check_add_fast(mask, u, v):
        """Check if adding edge (u, v) creates a violation. Only check triples involving u or v."""
        mu = mask[u]
        mv = mask[v]
        
        # Temporarily add edge
        mask[u] = mu | (1 << v)
        mask[v] = mv | (1 << u)
        
        valid = True
        # Check all triples that include u or v
        for w in range(n):
            if w == u or w == v:
                continue
            mw = mask[w]
            # Triple {u, v, w}
            if (mask[u] & mask[v] & mw).bit_count() >= 3:
                valid = False
                break
            for z in range(w + 1, n):
                if z == u or z == v:
                    continue
                mz = mask[z]
                # Triple {u, w, z}
                if (mask[u] & mw & mz).bit_count() >= 3:
                    valid = False
                    break
                # Triple {v, w, z}
                if (mask[v] & mw & mz).bit_count() >= 3:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            mask[u] = mu
            mask[v] = mv
            return False
        
        return True

    def repair(mask):
        """Remove edges to eliminate all violations. Prioritize edges in most violations."""
        max_iter = 5000
        for _ in range(max_iter):
            if time.time() > deadline:
                return False
            viols = find_violations(mask)
            if not viols:
                return True
            
            # Count how many times each edge appears in violations
            edge_viol_count = {}
            for i, j, k in viols:
                # For each violation triple, the edges that could fix it are those
                # incident to i, j, or k that go to common neighbors
                # Actually, removing any edge from {i,j,k} to a common neighbor helps
                # But simpler: remove an edge among the triple or to a common neighbor
                for v in range(n):
                    if v not in (i, j, k) and mask[i] & (1 << v) and mask[j] & (1 << v) and mask[k] & (1 << v):
                        key = (min(i, v), max(i, v))
                        edge_viol_count[key] = edge_viol_count.get(key, 0) + 1
                        key = (min(j, v), max(j, v))
                        edge_viol_count[key] = edge_viol_count.get(key, 0) + 1
                        key = (min(k, v), max(k, v))
                        edge_viol_count[key] = edge_viol_count.get(key, 0) + 1
            
            if not edge_viol_count:
                # Fallback: remove a random edge from one of the violation vertices
                i, j, k = viols[0]
                candidates = []
                for v in range(n):
                    if v != i and mask[i] & (1 << v):
                        candidates.append((i, v))
                    if v != j and mask[j] & (1 << v):
                        candidates.append((j, v))
                    if v != k and mask[k] & (1 << v):
                        candidates.append((k, v))
                if not candidates:
                    return False
                u, v = random.choice(candidates)
                mask[u] &= ~(1 << v)
                mask[v] &= ~(1 << u)
                continue
            
            # Remove the edge that appears in the most violations
            worst_edge = max(edge_viol_count, key=edge_viol_count.get)
            u, v = worst_edge
            mask[u] &= ~(1 << v)
            mask[v] &= ~(1 << u)
        
        return not find_violations(mask)

    def add_edges(mask):
        """Try to add edges one by one."""
        improved = True
        while improved:
            if time.time() > deadline:
                return
            improved = False
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not (mask[i] & (1 << j))]
            random.shuffle(edges)
            for u, v in edges:
                if time.time() > deadline:
                    return
                if check_add_fast(mask, u, v):
                    improved = True

    def local_search(mask):
        """Try edge swaps and removals to improve."""
        if time.time() > deadline:
            return False
        current_count = edge_count(mask)
        
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i] & (1 << j):
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        
        if not edges or not non_edges:
            return False
            
        random.shuffle(edges)
        random.shuffle(non_edges)
        
        for _ in range(100):
            if time.time() > deadline:
                return False
            a, b = random.choice(edges)
            c, d = random.choice(non_edges)
            
            mask[a] &= ~(1 << b)
            mask[b] &= ~(1 << a)
            
            if check_add_fast(mask, c, d):
                return True
            else:
                mask[a] |= (1 << b)
                mask[b] |= (1 << a)
        
        return False

    def deterministic_base():
        """Create a dense deterministic base graph."""
        mask = [0] * n
        # Start with complete graph
        for i in range(n):
            for j in range(i + 1, n):
                mask[i] |= (1 << j)
                mask[j] |= (1 << i)
        return mask

    def random_base(p):
        """Create a random base graph with edge probability p."""
        mask = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    mask[i] |= (1 << j)
                    mask[j] |= (1 << i)
        return mask

    best_mask = None
    best_count = -1
    random.seed(42)
    
    # Strategy 1: Start from complete graph and remove minimal edges
    mask = deterministic_base()
    if repair(mask):
        add_edges(mask)
        swap_improved = True
        while swap_improved and time.time() < deadline:
            swap_improved = local_search(mask)
        ec = edge_count(mask)
        if ec > best_count:
            best_count = ec
            best_mask = mask[:]
    
    # Strategy 2: Random restarts with various densities
    densities = [0.5, 0.6, 0.55, 0.45, 0.65, 0.4, 0.7]
    
    for p in densities:
        if time.time() > deadline:
            break
        
        for restart in range(20):
            if time.time() > deadline:
                break
            
            mask = random_base(p)
            if repair(mask):
                add_edges(mask)
                swap_improved = True
                while swap_improved and time.time() < deadline:
                    swap_improved = local_search(mask)
                
                ec = edge_count(mask)
                if ec > best_count:
                    best_count = ec
                    best_mask = mask[:]

    if best_mask is None:
        return []
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_mask[i] & (1 << j):
                edges.append([i, j])
    
    return edges
