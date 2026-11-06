import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # ---------- bitset helpers ----------
    def recompute(adj):
        """Compute N1, N2, N4 bitsets for each vertex."""
        N1 = adj[:]
        N2 = [0] * n
        for i in range(n):
            mask = 0
            x = N1[i]
            while x:
                lsb = x & -x
                j = lsb.bit_length() - 1
                mask |= N1[j]
                x ^= lsb
            N2[i] = mask
        N4 = [0] * n
        for i in range(n):
            mask = 0
            x = N2[i]
            while x:
                lsb = x & -x
                j = lsb.bit_length() - 1
                mask |= N2[j]
                x ^= lsb
        N4[i] = mask
        return N1, N2, N4

    def is_c6_free(adj):
        """Check if graph has no C6 using N1/N4 intersection test."""
        N1, N2, N4 = recompute(adj)
        for u in range(n):
            x = adj[u]
            while x:
                lsb = x & -x
                v = lsb.bit_length() - 1
                if u < v:
                    if (N1[u] & N4[v]) != 0 or (N1[v] & N4[u]) != 0:
                        return False
                x ^= lsb
        return True

    def add_edge_safe(u, v, adj, N1, N4):
        """Try to add edge (u,v) if it doesn't create a C6."""
        if adj[u] & (1 << v):
            return False
        if (N1[u] & N4[v]) != 0 or (N1[v] & N4[u]) != 0:
            return False
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        return True

    def edge_count(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2

    def to_edges(adj):
        edges = []
        for u in range(n):
            x = adj[u]
            while x:
                lsb = x & -x
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                x ^= lsb
        return edges

    # ---------- spectral / structured seed ----------
    def make_paley_like_seed():
        """Construct a seed graph inspired by Paley-type structure for n=40.
        We use a 5x8 grid with a quasi-random connection pattern that tends
        to have controlled local structure and suppresses long cycles."""
        adj = [0] * n
        # Arrange vertices as 8 rows of 5 (or 5 rows of 8). Use 8x5.
        # Within each row of 5: connect as K5 (10 edges per row * 8 = 80)
        # Between rows: use a deterministic pseudo-random pattern
        # Connect vertex (r, c) to vertex (r', c') if some condition holds.
        
        # Start with 8 disjoint K5's
        for r in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * r + a
                    v = 5 * r + b
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        
        # Add inter-row edges using a structured pattern.
        # For each pair of rows (r, r'), connect based on a permutation.
        # Use a simple rule: connect (r, c) to (r', c + shift[r][r']) mod 5
        # with some additional connections to get ~25 more edges.
        
        # Deterministic shifts based on row difference
        shifts = {}
        for dr in range(1, 8):
            # Use a permutation for each distance
            # For dr=1: shift by 1
            # For dr=2: shift by 2
            # etc., but also add a few extra edges
            shifts[dr] = dr % 5
        
        added = 0
        for r in range(8):
            for rp in range(r + 1, 8):
                dr = rp - r
                shift = shifts[dr]
                for c in range(5):
                    cp = (c + shift) % 5
                    u = 5 * r + c
                    v = 5 * rp + cp
                    if not (adj[u] & (1 << v)):
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                        added += 1
                    # Add a second connection for some pairs
                    if dr <= 2:
                        cp2 = (c + shift + 1) % 5
                        u2 = 5 * r + c
                        v2 = 5 * rp + cp2
                        if not (adj[u2] & (1 << v2)):
                            adj[u2] |= 1 << v2
                            adj[v2] |= 1 << u2
                            added += 1
        
        return adj

    def make_random_seed():
        """Random seed with moderate density."""
        adj = [0] * n
        for _ in range(50):
            u, v = random.sample(range(n), 2)
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        return adj

    def make_kronecker_seed():
        """8xK5 seed (proven 100 edges)."""
        adj = [0] * n
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
        return adj

    # ---------- greedy fill ----------
    def greedy_fill(adj):
        """Greedily add all safe edges."""
        N1, N2, N4 = recompute(adj)
        pairs = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    pairs.append((u, v))
        random.shuffle(pairs)
        for u, v in pairs:
            if add_edge_safe(u, v, adj, N1, N4):
                N1, N2, N4 = recompute(adj)
        return adj

    # ---------- local search: 1-swap (remove one, add one) ----------
    def local_search_1swap(adj, max_rounds=10):
        """Try to improve by removing one edge and adding another."""
        N1, N2, N4 = recompute(adj)
        count = edge_count(adj)
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            edges_list = to_edges(adj)
            random.shuffle(edges_list)
            for e1 in edges_list:
                if time.time() > deadline:
                    break
                u1, v1 = e1[0], e1[1]
                # Try to find a safe edge to add after removing (u1,v1)
                adj[u1] ^= (1 << v1)
                adj[v1] ^= (1 << u1)
                N1, N2, N4 = recompute(adj)
                
                candidates = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if not (adj[u] & (1 << v)):
                            if (N1[u] & N4[v]) == 0 and (N1[v] & N4[u]) == 0:
                                candidates.append((u, v))
                
                if candidates:
                    random.shuffle(candidates)
                    for u2, v2 in candidates[:20]:
                        adj[u2] |= 1 << v2
                        adj[v2] |= 1 << u2
                        # Verify C6-free
                        if is_c6_free(adj):
                            improved = True
                            count += 1
                            N1, N2, N4 = recompute(adj)
                            break
                        else:
                            adj[u2] ^= (1 << v2)
                            adj[v2] ^= (1 << u2)
                
                if not improved:
                    # Revert the removal
                    adj[u1] |= 1 << v1
                    adj[v1] |= 1 << u1
                    N1, N2, N4 = recompute(adj)
                if improved:
                    break
        
        return adj, count

    # ---------- local search: 2-swap (remove one, add two) ----------
    def local_search_2swap(adj):
        """Try to improve by removing one edge and adding two."""
        N1, N2, N4 = recompute(adj)
        count = edge_count(adj)
        
        edges_list = to_edges(adj)
        random.shuffle(edges_list)
        
        for e1 in edges_list:
            if time.time() > deadline:
                break
            u1, v1 = e1[0], e1[1]
            
            # Remove (u1, v1)
            adj[u1] ^= (1 << v1)
            adj[v1] ^= (1 << u1)
            N1, N2, N4 = recompute(adj)
            
            # Find candidate edges to add
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if (N1[u] & N4[v]) == 0 and (N1[v] & N4[u]) == 0:
                            candidates.append((u, v))
            
            if len(candidates) >= 2:
                random.shuffle(candidates)
                cand_subset = candidates[:30]
                for i in range(len(cand_subset)):
                    if time.time() > deadline:
                        break
                    u2, v2 = cand_subset[i]
                    adj[u2] |= 1 << v2
                    adj[v2] |= 1 << u2
                    N1, N2, N4 = recompute(adj)
                    
                    # Now try to add a second edge
                    for j in range(i + 1, len(cand_subset)):
                        u3, v3 = cand_subset[j]
                        if adj[u3] & (1 << v3):
                            continue
                        if (N1[u3] & N4[v3]) != 0 or (N1[v3] & N4[u3]) != 0:
                            continue
                        adj[u3] |= 1 << v3
                        adj[v3] |= 1 << u3
                        
                        if is_c6_free(adj):
                            count += 2
                            N1, N2, N4 = recompute(adj)
                            # Revert the first addition's effect is already included
                            # We've successfully done the 2-swap
                            adj[u1] |= 1 << v1  # don't re-add the removed edge
                            adj[v1] |= 1 << u1
                            # Wait - we removed (u1,v1) and added two edges. That's net +1.
                            # Actually we should NOT re-add (u1,v1). The 2-swap means
                            # remove 1, add 2, net gain 1.
                            adj[u1] ^= (1 << v1)
                            adj[v1] ^= (1 << u1)
                            return adj, count, True
                        else:
                            adj[u3] ^= (1 << v3)
                            adj[v3] ^= (1 << u3)
                            N1, N2, N4 = recompute(adj)
                    
                    # Revert second edge
                    adj[u2] ^= (1 << v2)
                    adj[v2] ^= (1 << u2)
                    N1, N2, N4 = recompute(adj)
            
            # Revert the removal
            adj[u1] |= 1 << v1
            adj[v1] |= 1 << u1
            N1, N2, N4 = recompute(adj)
        
        return adj, count, False

    # ---------- main search loop ----------
    best_edges = []
    best_count = 0

    def evaluate_and_update(adj):
        nonlocal best_edges, best_count
        if is_c6_free(adj):
            c = edge_count(adj)
            if c > best_count:
                best_count = c
                best_edges = to_edges(adj)

    def run_search(adj):
        nonlocal best_edges, best_count
        
        # Greedy fill
        adj = greedy_fill(adj)
        evaluate_and_update(adj)
        
        # Local search 1-swap
        adj, count = local_search_1swap(adj, max_rounds=5)
        evaluate_and_update(adj)
        
        # Local search 2-swap (a few attempts)
        for _ in range(3):
            if time.time() > deadline:
                break
            adj, count, improved = local_search_2swap(adj)
            if not improved:
                break
            evaluate_and_update(adj)
        
        return adj

    # Multiple restarts with different seeds
    seeds = [make_kronecker_seed(), make_paley_like_seed()]
    
    # Add some randomized structured seeds
    for _ in range(2):
        adj = make_kronecker_seed()
        # Randomly perturb: remove some edges, add some random ones
        for _ in range(random.randint(5, 15)):
            u, v = random.sample(range(n), 2)
            if adj[u] & (1 << v):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= 1 << v
                adj[v] |= 1 << u
        seeds.append(adj)
    
    seeds.append(make_random_seed())
    
    random.shuffle(seeds)
    
    for seed_adj in seeds:
        if time.time() > deadline:
            break
        adj = seed_adj[:]
        run_search(adj)
    
    # More restarts with random perturbations of the best found
    while time.time() < deadline:
        if best_edges:
            adj = [0] * n
            for u, v in best_edges:
                adj[u] |= 1 << v
                adj[v] |= 1 << u
            # Perturb
            for _ in range(random.randint(3, 10)):
                u, v = random.sample(range(n), 2)
                if adj[u] & (1 << v):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
            run_search(adj)
        else:
            adj = make_random_seed()
            run_search(adj)
    
    # If we still have time, do more structured restarts
    while time.time() < deadline:
        adj = make_kronecker_seed()
        # Add random inter-block edges
        for _ in range(random.randint(5, 20)):
            r1, r2 = random.sample(range(8), 2)
            c1 = random.randint(0, 4)
            c2 = random.randint(0, 4)
            u = 5 * r1 + c1
            v = 5 * r2 + c2
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        run_search(adj)
    
    return best_edges
