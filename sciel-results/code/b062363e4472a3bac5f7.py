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

    def polarity_base():
        """
        Create a base graph inspired by polarity graphs of projective planes.
        For n=20, we use a structure based on finite geometry to get a dense
        K3,3-free starting point. We use a 5x4 grid structure with specific
        adjacency rules to avoid K3,3.
        
        Vertices are (r, c) for r in 0..3, c in 0..4 (20 vertices).
        Two vertices (r1, c1) and (r2, c2) are adjacent if:
        - r1 == r2 and c1 != c2 (same row)
        - c1 == c2 and r1 != r2 (same column)
        - (r1 + r2) % 5 == 0 and (c1 + c2) % 4 == 0 (polarity-like)
        - |r1 - r2| == 1 and |c1 - c2| == 1 (diagonal neighbors)
        
        This creates a structured dense graph that may have fewer K3,3 subgraphs
        than a random graph of similar density.
        """
        mask = [0] * n
        
        def idx(r, c):
            return r * 4 + c
        
        # Add edges based on rules
        for r1 in range(4):
            for c1 in range(4):
                for r2 in range(r1, 4):
                    for c2 in range(4 if r1 == r2 else 0, 4):
                        if r1 == r2 and c1 == c2:
                            continue
                        u = idx(r1, c1)
                        v = idx(r2, c2)
                        
                        should_add = False
                        # Same row
                        if r1 == r2 and c1 != c2:
                            should_add = True
                        # Same column
                        elif c1 == c2 and r1 != r2:
                            should_add = True
                        # Polarity-like: (r1+r2) % 5 == 0 and (c1+c2) % 4 == 0
                        elif (r1 + r2) % 5 == 0 and (c1 + c2) % 4 == 0:
                            should_add = True
                        # Diagonal neighbors
                        elif abs(r1 - r2) == 1 and abs(c1 - c2) == 1:
                            should_add = True
                        # Additional: wrap-around row adjacency
                        elif abs(r1 - r2) == 1 and c1 == c2:
                            should_add = True
                        elif r1 == r2 and abs(c1 - c2) == 1:
                            should_add = True
                        # Extra edges to increase density
                        elif (r1 + 2 * c1) % 4 == (r2 + 2 * c2) % 4:
                            should_add = True
                        elif (c1 + 3 * r1) % 5 == (c2 + 3 * r2) % 5:
                            should_add = True
                        
                        if should_add:
                            mask[u] |= (1 << v)
                            mask[v] |= (1 << u)
        
        return mask

    def structured_base():
        """
        Create a structured base graph using a different geometric construction.
        Uses a 4x5 grid with toroidal adjacency and additional structured edges.
        """
        mask = [0] * n
        
        def idx(r, c):
            return r * 5 + c
        
        for r1 in range(4):
            for c1 in range(5):
                for r2 in range(r1, 4):
                    for c2 in range(5 if r1 == r2 else 0, 5):
                        if r1 == r2 and c1 == c2:
                            continue
                        u = idx(r1, c1)
                        v = idx(r2, c2)
                        
                        should_add = False
                        # Toroidal row neighbors
                        if r1 == r2 and min(abs(c1 - c2), 5 - abs(c1 - c2)) <= 2:
                            should_add = True
                        # Toroidal column neighbors
                        elif c1 == c2 and min(abs(r1 - r2), 4 - abs(r1 - r2)) <= 1:
                            should_add = True
                        # Diagonal toroidal
                        elif min(abs(r1 - r2), 4 - abs(r1 - r2)) <= 1 and min(abs(c1 - c2), 5 - abs(c1 - c2)) <= 1:
                            if not (r1 == r2 and c1 == c2):
                                should_add = True
                        # Structured cross edges
                        elif (r1 + c1) % 3 == (r2 + c2) % 3 and (r1 - c1) % 2 == (r2 - c2) % 2:
                            should_add = True
                        elif (r1 * 2 + c1) % 4 == (r2 * 2 + c2) % 4:
                            should_add = True
                        
                        if should_add:
                            mask[u] |= (1 << v)
                            mask[v] |= (1 << u)
        
        return mask

    def refine_mask(mask):
        """Apply repair, add_edges, and local_search to a mask."""
        if not repair(mask):
            return False
        add_edges(mask)
        swap_improved = True
        while swap_improved and time.time() < deadline:
            swap_improved = local_search(mask)
        return True

    best_mask = None
    best_count = -1
    random.seed(42)
    
    # Strategy 1: Start from complete graph and remove minimal edges
    mask = deterministic_base()
    if refine_mask(mask):
        ec = edge_count(mask)
        if ec > best_count:
            best_count = ec
            best_mask = mask[:]
    
    # Strategy 2: Polarity-inspired base
    mask = polarity_base()
    if refine_mask(mask):
        ec = edge_count(mask)
        if ec > best_count:
            best_count = ec
            best_mask = mask[:]
    
    # Strategy 3: Structured base
    mask = structured_base()
    if refine_mask(mask):
        ec = edge_count(mask)
        if ec > best_count:
            best_count = ec
            best_mask = mask[:]
    
    # Strategy 4: Random restarts with various densities
    densities = [0.5, 0.6, 0.55, 0.45, 0.65, 0.4, 0.7, 0.52, 0.58]
    
    for p in densities:
        if time.time() > deadline:
            break
        
        for restart in range(15):
            if time.time() > deadline:
                break
            
            mask = random_base(p)
            if refine_mask(mask):
                ec = edge_count(mask)
                if ec > best_count:
                    best_count = ec
                    best_mask = mask[:]
    
    # Strategy 5: Perturb best found and refine
    if best_mask is not None and time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            mask = best_mask[:]
            # Randomly remove some edges
            for i in range(n):
                for j in range(i + 1, n):
                    if mask[i] & (1 << j) and random.random() < 0.1:
                        mask[i] &= ~(1 << j)
                        mask[j] &= ~(1 << i)
            if refine_mask(mask):
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
