"""Combined approach: Spectral/structured initialization + refined local search for K4-free graph optimization."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
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
            
            # Option 1: Exclude v
            backtrack(idx + 1, current_mask, current_size)
            
            # Option 2: Include v
            if not timeout[0]:
                can_include = True
                N_v_mask = adj[v]
                C = current_mask & N_v_mask
                
                tmp_c = C
                valid = True
                while tmp_c:
                    bit = tmp_c & (-tmp_c)
                    u = bit.bit_length() - 1
                    tmp_c ^= bit
                    if adj[u] & C:
                        valid = False
                        break
                
                if valid:
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
    
    def construct_paley_17():
        """Construct Paley(17) graph on vertices 0..16, and vertex 17 isolated."""
        adj = [0] * n
        residues = {1, 4, 9, 16, 8, 2, 15, 13}
        
        for i in range(17):
            for j in range(i + 1, 17):
                diff = (j - i) % 17
                if diff in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Vertex 17 is isolated
        return adj
    
    def construct_clebsch_complement():
        """
        Construct the complement of the Clebsch graph (16 vertices) + 2 isolated vertices.
        Clebsch graph is strongly regular (16, 5, 0, 2).
        Complement is strongly regular (16, 10, 6, 6).
        This graph is K4-free? 
        Wait, Clebsch graph is K4-free. Its complement might have K4s.
        Let's check: In Clebsch, triangles are 0. In complement, triangles correspond to independent sets of size 3 in Clebsch.
        Clebsch has alpha=2? No, alpha(Clebsch)=2? 
        Actually, let's just use a random regular graph with spectral properties or stick to Paley + perturbations.
        The prompt suggests "complement of the Clebsch graph". 
        Clebsch graph vertices can be represented as binary strings of length 4 with even parity.
        Two vertices are adjacent if Hamming distance is 2.
        Complement: adjacent if Hamming distance is 4.
        Distance 4 in Clebsch means the vertices are disjoint in the even parity set?
        Let's just generate a random 10-regular graph on 16 vertices and check K4-freeness, or use the Paley one which is known to be good.
        Actually, Paley(17) is 8-regular. Complement is 8-regular.
        Let's try to construct a graph based on the Clebsch complement structure if possible, or just use Paley as the primary "structured" seed.
        """
        adj = [0] * n
        # Vertices 0-15 form the Clebsch complement, 16-17 isolated?
        # Clebsch vertices: 0..15
        # Adjacency in Clebsch: dist=2. Complement: dist=4.
        # In 4-bit space, dist 4 means bitwise complement.
        # So in complement of Clebsch, i is connected to ~i (bitwise not) ?
        # No, Hamming distance 4 means all bits differ.
        # So each vertex is connected to exactly 1 other vertex (its complement).
        # That's a perfect matching. That's not strongly regular (16,10,6,6).
        # My memory of Clebsch parameters might be off or I'm confusing it.
        # Clebsch: v=16, k=5, lambda=0, mu=2.
        # Complement: v=16, k=10, lambda=6, mu=6.
        # Let's just use Paley(17) as it's robust and known.
        return construct_paley_17()
    
    def edge_swap(adj, u, v, w, x):
        """Perform edge swap: remove (u,v) and (w,x), add (u,w) and (v,x)."""
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
    
    # Strategy 1: Paley(17) + isolated vertex
    adj_paley = construct_paley_17()
    if make_k4_free(adj_paley):
        candidates.append(adj_paley[:])
    
    # Strategy 2: Random perturbations of Paley(17)
    for trial in range(50):
        if time.time() > deadline - 50:
            break
        adj = adj_paley[:]
        # Flip 1-3 random edges
        num_flips = random.randint(1, 3)
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
        
        if make_k4_free(adj):
            candidates.append(adj[:])
            
    # Strategy 3: Tripartite-like structures (inspiration from K6,6,6 but randomized)
    for trial in range(20):
        if time.time() > deadline - 50:
            break
        adj = [0] * n
        # Random 3-partition
        parts = [[], [], []]
        for v in range(n):
            parts[random.randint(0, 2)].append(v)
        
        # Add edges between parts with high probability, within with low
        for i in range(3):
            for j in range(i + 1, 3):
                for u in parts[i]:
                    for v in parts[j]:
                        if random.random() < 0.8:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
            # Within part edges (sparse)
            for u in parts[i]:
                for v in parts[i]:
                    if u < v and random.random() < 0.1:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        if make_k4_free(adj):
            candidates.append(adj[:])

    # Evaluate all candidates
    best_adj = None
    best_score = 18
    for adj in candidates:
        if time.time() > deadline - 30:
            break
        # Use a tighter time limit for initial screening
        score = max_triangle_free_size(adj, time_limit=2.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
    
    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(best_adj, time_limit=2.0)
    
    # Local search with simulated annealing
    adj = best_adj[:]
    current_score = best_score
    best_adj_sa = adj[:]
    best_score_sa = current_score
    
    temperature = 1.0
    cooling = 0.995
    max_iter = 8000
    
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
        
        move_type = random.random()
        
        if move_type < 0.5:
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
            
            if has_k4(adj):
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
        
        elif move_type < 0.9:
            # Edge swap
            edges = get_edges(adj)
            if len(edges) < 2:
                continue
            
            e1 = random.choice(edges)
            e2 = random.choice(edges)
            u, v = e1
            w, x = e2
            
            old_adj = adj[:]
            if edge_swap(adj, u, v, w, x):
                if not has_k4(adj):
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
                else:
                    adj = old_adj
            else:
                adj = old_adj
        else:
            # Multi-edge flip
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
            
            if has_k4(adj):
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
