"""Combined approach: spectral initialization + refined local search for K4-free graph optimization."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    n = 18
    
    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    def has_k4(adj):
        """Check if graph contains a K4 using bitmask operations."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    d_candidates = common & adj[c]
                    d_candidates &= ~((1 << (c + 1)) - 1)
                    if d_candidates:
                        return True
        return False
    
    def find_k4(adj):
        """Find a K4 in the graph, return tuple (a,b,c,d) or None."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    d_candidates = common & adj[c]
                    d_candidates &= ~((1 << (c + 1)) - 1)
                    if d_candidates:
                        d_bit = d_candidates & (-d_candidates)
                        d = d_bit.bit_length() - 1
                        return (a, b, c, d)
        return None
    
    def max_triangle_free_size(adj, time_limit=5.0):
        """Find largest induced triangle-free subgraph using branch and bound with pruning."""
        best = [0]
        t0 = time.time()
        timeout = [False]
        
        neigh = adj[:]
        
        # Precompute conflict lists for faster triangle detection
        conflict_lists = [[] for _ in range(n)]
        for v in range(n):
            neigh_v = neigh[v]
            neighbors_v = []
            tmp = neigh_v
            while tmp:
                bit = tmp & (-tmp)
                u = bit.bit_length() - 1
                neighbors_v.append(u)
                tmp ^= bit
            
            for u in neighbors_v:
                conflict_mask = neigh_v & neigh[u] & ~(1 << u) & ~(1 << v)
                if conflict_mask:
                    conflict_lists[v].append((u, conflict_mask))
        
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])
        
        def backtrack(idx, current_mask, current_size):
            if timeout[0]:
                return
            if time.time() - t0 > time_limit:
                timeout[0] = True
                return
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            if current_size + (n - idx) <= best[0]:
                return
            
            v = order[idx]
            
            # Option 1: exclude v
            backtrack(idx + 1, current_mask, current_size)
            
            # Option 2: include v
            if not timeout[0]:
                can_include = True
                for u, conflict_mask in conflict_lists[v]:
                    if (current_mask & (1 << u)) and (current_mask & conflict_mask):
                        can_include = False
                        break
                
                if can_include:
                    backtrack(idx + 1, current_mask | (1 << v), current_size + 1)
        
        backtrack(0, 0, 0)
        return best[0]
    
    def make_k4_free(adj):
        """Remove edges to make graph K4-free. Returns True if successful."""
        for _ in range(500):
            k4 = find_k4(adj)
            if k4 is None:
                return True
            a, b, c, d = k4
            # Remove a random edge from the K4
            pair = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
            x, y = pair
            adj[x] &= ~(1 << y)
            adj[y] &= ~(1 << x)
        return not has_k4(adj)
    
    def random_regular_graph(n, degree, seed=None):
        """Generate a random regular graph using configuration model."""
        if seed is not None:
            random.seed(seed)
        adj = [0] * n
        # Create stub list
        stubs = []
        for i in range(n):
            for _ in range(degree):
                stubs.append(i)
        random.shuffle(stubs)
        
        # Pair up stubs
        i = 0
        while i < len(stubs):
            a = stubs[i]
            j = i + 1
            while j < len(stubs) and stubs[j] == a:
                j += 1
            if j >= len(stubs):
                return None  # Failed
            b = stubs[j]
            if a == b:
                return None  # Self-loop
            if (adj[a] >> b) & 1:
                return None  # Multi-edge
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
            i = j + 1
        
        # Verify regularity
        for v in range(n):
            if bin(adj[v]).count('1') != degree:
                return None
        return adj
    
    def count_triangles(adj):
        """Count triangles in the graph."""
        count = 0
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                common &= ~((1 << (b + 1)) - 1)
                count += bin(common).count('1')
        return count
    
    def edge_swap(adj, u, v, w, x):
        """Perform edge swap: remove (u,v) and (w,x), add (u,w) and (v,x)."""
        # Check preconditions
        if u == v or w == x:
            return False
        if u == w or u == x or v == w or v == x:
            return False
        if not ((adj[u] >> v) & 1):
            return False
        if not ((adj[w] >> x) & 1):
            return False
        if (adj[u] >> w) & 1:
            return False
        if (adj[v] >> x) & 1:
            return False
        
        # Perform swap
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        adj[w] &= ~(1 << x)
        adj[x] &= ~(1 << w)
        adj[u] |= (1 << w)
        adj[w] |= (1 << u)
        adj[v] |= (1 << x)
        adj[x] |= (1 << v)
        return True
    
    random.seed(42)
    candidates = []
    
    # Strategy 1: Random regular graphs with various degrees
    for degree in [6, 7, 8, 9]:
        for trial in range(15):
            if time.time() > deadline - 50:
                break
            adj = random_regular_graph(n, degree)
            if adj is not None:
                if make_k4_free(adj):
                    candidates.append(adj[:])
    
    # Strategy 2: Complete 3-partite K_{6,6,6}
    adj1 = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj1[a] |= (1 << b)
                    adj1[b] |= (1 << a)
    candidates.append(adj1[:])
    
    # Strategy 3: Random graphs with controlled density
    for trial in range(30):
        if time.time() > deadline - 50:
            break
        density = 0.5 + random.random() * 0.3
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if make_k4_free(adj):
            candidates.append(adj[:])
    
    # Strategy 4: Blow-up of small K4-free graphs
    for trial in range(20):
        if time.time() > deadline - 50:
            break
        # Random K4-free graph on 6 vertices
        small_adj = [0] * 6
        for i in range(6):
            for j in range(i + 1, 6):
                if random.random() < 0.5:
                    small_adj[i] |= (1 << j)
                    small_adj[j] |= (1 << i)
        
        # Check if K4-free
        is_k4_free = True
        for a in range(6):
            for b in range(a + 1, 6):
                if not ((small_adj[a] >> b) & 1):
                    continue
                common = small_adj[a] & small_adj[b]
                for c in range(b + 1, 6):
                    if not ((common >> c) & 1):
                        continue
                    d_candidates = common & small_adj[c]
                    d_candidates &= ~((1 << (c + 1)) - 1)
                    if d_candidates:
                        is_k4_free = False
                        break
                if not is_k4_free:
                    break
            if not is_k4_free:
                break
        if not is_k4_free:
            continue
        
        # Blow up: each vertex i becomes vertices 3*i, 3*i+1, 3*i+2
        adj = [0] * n
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                if (small_adj[i] >> j) & 1:
                    for a in range(3):
                        for b in range(3):
                            u = 3 * i + a
                            v = 3 * j + b
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        # Add random within-blowup edges
        for i in range(6):
            for a in range(3):
                for b in range(a + 1, 3):
                    if random.random() < 0.3:
                        u = 3 * i + a
                        v = 3 * i + b
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        if make_k4_free(adj):
            candidates.append(adj[:])
    
    # Strategy 5: Structured graphs based on finite fields
    # Use a 3x6 grid structure with specific connection patterns
    for trial in range(15):
        if time.time() > deadline - 50:
            break
        adj = [0] * n
        # Create a 3-partite structure with varying cross densities
        part1 = list(range(0, 6))
        part2 = list(range(6, 12))
        part3 = list(range(12, 18))
        
        # Random permutation within parts
        random.shuffle(part1)
        random.shuffle(part2)
        random.shuffle(part3)
        
        # Connect with varying densities
        for a in part1:
            for b in part2:
                if random.random() < 0.7:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
        for a in part2:
            for b in part3:
                if random.random() < 0.7:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
        for a in part1:
            for b in part3:
                if random.random() < 0.7:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
        
        # Add some within-part edges
        for part in [part1, part2, part3]:
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    if random.random() < 0.2:
                        adj[part[i]] |= (1 << part[j])
                        adj[part[j]] |= (1 << part[i])
        
        if make_k4_free(adj):
            candidates.append(adj[:])
    
    # Evaluate all candidates
    best_adj = None
    best_score = 18
    for adj in candidates:
        if time.time() > deadline - 30:
            break
        score = max_triangle_free_size(adj, time_limit=2.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
    
    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(best_adj, time_limit=2.0)
    
    # Local search with simulated annealing and edge swaps
    adj = best_adj[:]
    current_score = best_score
    best_adj_sa = adj[:]
    best_score_sa = current_score
    
    temperature = 1.0
    cooling = 0.995
    max_iter = 8000
    
    # Precompute edge list for swap operations
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
        return edges
    
    for iteration in range(max_iter):
        if time.time() > deadline - 15:
            break
        if best_score_sa <= 6:
            break
        
        # Choose move type: single edge flip or edge swap
        move_type = random.random()
        
        if move_type < 0.4:
            # Single edge flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_adj = adj[:]
            if (adj[i] >> j) & 1:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
            else:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            
            k4 = find_k4(adj)
            if k4 is not None:
                adj = old_adj
                continue
            
            score = max_triangle_free_size(adj, time_limit=1.5)
            delta = score - current_score
            
            accept = False
            if delta <= 0:
                accept = True
            else:
                exp_val = math.exp(-delta / max(temperature, 0.01))
                if random.random() < exp_val:
                    accept = True
            
            if accept:
                current_score = score
                if score < best_score_sa:
                    best_score_sa = score
                    best_adj_sa = adj[:]
            else:
                adj = old_adj
        
        elif move_type < 0.8:
            # Edge swap (2-switch)
            edges = get_edges(adj)
            if len(edges) < 2:
                continue
            
            # Try multiple random swaps
            for _ in range(3):
                if time.time() > deadline - 15:
                    break
                
                e1 = random.choice(edges)
                e2 = random.choice(edges)
                u, v = e1
                w, x = e2
                
                old_adj = adj[:]
                if edge_swap(adj, u, v, w, x):
                    k4 = find_k4(adj)
                    if k4 is None:
                        score = max_triangle_free_size(adj, time_limit=1.5)
                        delta = score - current_score
                        
                        accept = False
                        if delta <= 0:
                            accept = True
                        else:
                            exp_val = math.exp(-delta / max(temperature, 0.01))
                            if random.random() < exp_val:
                                accept = True
                        
                        if accept:
                            current_score = score
                            if score < best_score_sa:
                                best_score_sa = score
                                best_adj_sa = adj[:]
                            break
                        else:
                            adj = old_adj
                    else:
                        adj = old_adj
                else:
                    adj = old_adj
        
        else:
            # Multi-edge move: flip 2-3 random edges
            num_flips = random.randint(2, 3)
            old_adj = adj[:]
            for _ in range(num_flips):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                if (adj[i] >> j) & 1:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            
            k4 = find_k4(adj)
            if k4 is not None:
                adj = old_adj
                continue
            
            score = max_triangle_free_size(adj, time_limit=1.5)
            delta = score - current_score
            
            accept = False
            if delta <= 0:
                accept = True
            else:
                exp_val = math.exp(-delta / max(temperature, 0.01))
                if random.random() < exp_val:
                    accept = True
            
            if accept:
                current_score = score
                if score < best_score_sa:
                    best_score_sa = score
                    best_adj_sa = adj[:]
            else:
                adj = old_adj
        
        temperature *= cooling
    
    # Final evaluation with more time
    final_score = max_triangle_free_size(best_adj_sa, time_limit=20.0)
    
    return edges_from_adj(best_adj_sa)
