"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

class BitGraph:
    """
    Maintains a graph using bitmasks for adjacency to allow fast triangle detection and scoring.
    """
    def __init__(self, n, edges):
        self.n = n
        self.adj = [0] * n
        self.deg = [0] * n
        self.edge_set = set()
        for u, v in edges:
            self.add_edge(u, v)
        # Precompute all pairs for fast non-edge sampling
        self.all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def add_edge(self, u, v):
        if u == v:
            return
        key = (min(u, v), max(u, v))
        if key in self.edge_set:
            return
        self.edge_set.add(key)
        self.adj[u] |= (1 << v)
        self.adj[v] |= (1 << u)
        self.deg[u] += 1
        self.deg[v] += 1

    def remove_edge(self, u, v):
        key = (min(u, v), max(u, v))
        if key not in self.edge_set:
            return
        self.edge_set.remove(key)
        self.adj[u] &= ~(1 << v)
        self.adj[v] &= ~(1 << u)
        self.deg[u] -= 1
        self.deg[v] -= 1

    def get_score(self):
        """
        Compute the maximum triangle degree sum.
        """
        max_sum = 0
        for u, v in self.edge_set:
            common = self.adj[u] & self.adj[v]
            while common:
                lsb = common & -common
                w = lsb.bit_length() - 1
                s = self.deg[u] + self.deg[v] + self.deg[w]
                if s > max_sum:
                    max_sum = s
                common ^= lsb
        return max_sum

    def get_worst_triangle(self):
        """
        Return the triangle (u, v, w) with the maximum degree sum, or None if no triangles.
        """
        max_sum = -1
        worst_tri = None
        for u, v in self.edge_set:
            common = self.adj[u] & self.adj[v]
            while common:
                lsb = common & -common
                w = lsb.bit_length() - 1
                s = self.deg[u] + self.deg[v] + self.deg[w]
                if s > max_sum:
                    max_sum = s
                    worst_tri = (u, v, w)
                common ^= lsb
        return worst_tri, max_sum

    def is_edge(self, u, v):
        return (min(u, v), max(u, v)) in self.edge_set

    def random_non_edge(self, exclude=None):
        """
        Pick a random non-edge, optionally excluding a specific pair.
        Returns (u, v) or None if no non-edges exist.
        """
        n = self.n
        # Use rejection sampling with a bounded number of tries
        for _ in range(50):
            u = random.randrange(n)
            v = random.randrange(n)
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in self.edge_set:
                continue
            if exclude is not None and (a, b) == exclude:
                continue
            return (a, b)
        # Fallback: iterate through all pairs
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in self.edge_set:
                    if exclude is not None and (i, j) == exclude:
                        continue
                    return (i, j)
        return None

    def get_non_edge_list(self):
        """Return list of all non-edges."""
        result = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if (i, j) not in self.edge_set:
                    result.append((i, j))
        return result

def _construct_baseline():
    """K_{8,8} plus one edge inside a part."""
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append((u, v))
    edges.append((0, 1))
    return edges

def _local_search(n, initial_edges, deadline):
    """
    Local search with simulated annealing and improved move generation.
    """
    start_time = time.time()
    
    # Initialize state
    g = BitGraph(n, initial_edges)
    current_score = g.get_score()
    best_score = current_score
    best_edges = set(g.edge_set)
    
    T = 4.0
    T_min = 0.005
    cooling = 0.9995
    
    iterations = 0
    no_improve_count = 0
    last_best_time = start_time
    
    while time.time() < deadline:
        iterations += 1
        if iterations % 500 == 0 and time.time() > deadline:
            break
        
        # 1. Identify worst triangle
        worst_tri, worst_val = g.get_worst_triangle()
        
        # 2. Choose edge to remove
        if worst_tri and worst_val > 0:
            u, v, w = worst_tri
            # With probability 0.8, remove an edge from the worst triangle
            if random.random() < 0.8:
                # Pick the edge between the two highest-degree vertices in the triangle
                degs = [(g.deg[u], u, v), (g.deg[v], v, w), (g.deg[w], w, u)]
                # Sort by max degree of the two endpoints
                degs.sort(key=lambda x: max(g.deg[x[1]], g.deg[x[2]]), reverse=True)
                rem_u, rem_v = degs[0][1], degs[0][2]
            else:
                # Remove a random edge
                rem_u, rem_v = random.choice(list(g.edge_set))
        else:
            rem_u, rem_v = random.choice(list(g.edge_set))
        
        # 3. Choose edge to add
        rem_key = (min(rem_u, rem_v), max(rem_u, rem_v))
        add_pair = g.random_non_edge(exclude=rem_key)
        
        if add_pair is None:
            continue
            
        add_u, add_v = add_pair
        
        # 4. Perform Swap
        g.remove_edge(rem_u, rem_v)
        g.add_edge(add_u, add_v)
        
        new_score = g.get_score()
        delta = new_score - current_score
        
        # 5. Accept/Reject
        accept = False
        if delta < 0:
            accept = True
        elif T > T_min and delta > 0:
            # Accept with probability exp(-delta/T)
            if random.random() < math.exp(-delta / T):
                accept = True
        
        if accept:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = set(g.edge_set)
                no_improve_count = 0
                last_best_time = time.time()
            else:
                no_improve_count += 1
        else:
            # Revert
            g.remove_edge(add_u, add_v)
            g.add_edge(rem_u, rem_v)
            no_improve_count += 1
        
        T *= cooling
        
        # Restart if stuck for a long time
        if no_improve_count > 2000:
            # Reset temperature and restart from best
            g = BitGraph(n, list(best_edges))
            current_score = best_score
            T = 3.0
            no_improve_count = 0

    return list(best_edges)

def _polish(g, deadline):
    """
    Final polish: only accept moves that strictly improve the score.
    """
    n = g.n
    current_score = g.get_score()
    best_score = current_score
    best_edges = set(g.edge_set)
    start_time = time.time()
    
    while time.time() < deadline:
        # Get worst triangle
        worst_tri, worst_val = g.get_worst_triangle()
        
        if worst_tri is None or worst_val == 0:
            break
        
        u, v, w = worst_tri
        # Try removing each edge of the worst triangle and adding a good non-edge
        edges_to_try = [(u, v), (v, w), (w, u)]
        random.shuffle(edges_to_try)
        
        improved = False
        for rem_u, rem_v in edges_to_try:
            if not g.is_edge(rem_u, rem_v):
                continue
            rem_key = (min(rem_u, rem_v), max(rem_u, rem_v))
            
            # Try several candidate non-edges
            for _ in range(10):
                add_pair = g.random_non_edge(exclude=rem_key)
                if add_pair is None:
                    break
                add_u, add_v = add_pair
                
                # Perform swap
                g.remove_edge(rem_u, rem_v)
                g.add_edge(add_u, add_v)
                
                new_score = g.get_score()
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_edges = set(g.edge_set)
                    improved = True
                else:
                    # Revert
                    g.remove_edge(add_u, add_v)
                    g.add_edge(rem_u, rem_v)
            
            if improved:
                break
        
        if not improved:
            # Try random moves
            rem_u, rem_v = random.choice(list(g.edge_set))
            rem_key = (min(rem_u, rem_v), max(rem_u, rem_v))
            add_pair = g.random_non_edge(exclude=rem_key)
            if add_pair is None:
                break
            add_u, add_v = add_pair
            
            g.remove_edge(rem_u, rem_v)
            g.add_edge(add_u, add_v)
            new_score = g.get_score()
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_edges = set(g.edge_set)
            else:
                g.remove_edge(add_u, add_v)
                g.add_edge(rem_u, rem_v)

    return list(best_edges)

def construct():
    """Construct a graph on 16 vertices with 65 edges minimizing max triangle degree sum."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n = 16
    target_edges = 65
    
    # Start with baseline
    initial_edges = _construct_baseline()
    
    # Run local search with most of the budget
    search_deadline = start_time + 95
    best_edges = _local_search(n, initial_edges, search_deadline)
    
    # Final polish phase with remaining time
    polish_deadline = start_time + 115
    g = BitGraph(n, best_edges)
    best_edges = _polish(g, polish_deadline)
    
    # Verify edge count
    edge_set = set()
    for u, v in best_edges:
        edge_set.add((min(u, v), max(u, v)))
        
    if len(edge_set) != target_edges:
        # Fallback
        best_edges = _construct_baseline()
        edge_set = set()
        for u, v in best_edges:
            edge_set.add((min(u, v), max(u, v)))
        
    return {'n': n, 'edges': [list(e) for e in sorted(edge_set)]}

# EVOLVE-BLOCK-END
