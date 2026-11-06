"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

class BitGraph:
    """
    Maintains a graph using bitmasks for adjacency to allow fast triangle detection and scoring.
    Optimized for n=16.
    """
    def __init__(self, n, edges):
        self.n = n
        self.adj = [0] * n
        self.deg = [0] * n
        self.edge_set = set()
        for u, v in edges:
            self.add_edge(u, v)
        
        # Precompute all pairs for fast non-edge sampling/iteration
        self.all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        # Store non-edges as a list for O(1) access to random non-edge if needed, 
        # but for n=16, iterating all_pairs is also fast. 
        # We'll use a helper to get a random non-edge.

    def add_edge(self, u, v):
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in self.edge_set:
            return False
        self.edge_set.add(key)
        self.adj[u] |= (1 << v)
        self.adj[v] |= (1 << u)
        self.deg[u] += 1
        self.deg[v] += 1
        return True

    def remove_edge(self, u, v):
        key = (min(u, v), max(u, v))
        if key not in self.edge_set:
            return False
        self.edge_set.remove(key)
        self.adj[u] &= ~(1 << v)
        self.adj[v] &= ~(1 << u)
        self.deg[u] -= 1
        self.deg[v] -= 1
        return True

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

    def get_random_non_edge(self, exclude=None):
        """
        Pick a random non-edge, optionally excluding a specific pair.
        """
        n = self.n
        # Rejection sampling is okay for n=16, but let's try to be smarter.
        # Since density is ~50%, rejection sampling takes ~2 tries on average.
        for _ in range(20):
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
        
        # Fallback: iterate
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in self.edge_set:
                    if exclude is not None and (i, j) == exclude:
                        continue
                    return (i, j)
        return None

    def get_all_non_edges(self):
        """Return list of all non-edges."""
        result = []
        for i, j in self.all_pairs:
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
    Local search with simulated annealing and targeted move generation.
    """
    start_time = time.time()
    
    # Initialize state
    g = BitGraph(n, initial_edges)
    current_score = g.get_score()
    best_score = current_score
    best_edges = set(g.edge_set)
    
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    iterations = 0
    no_improve_count = 0
    last_best_time = start_time
    
    while time.time() < deadline:
        iterations += 1
        
        # 1. Identify worst triangle
        worst_tri, worst_val = g.get_worst_triangle()
        
        # 2. Choose edge to remove
        if worst_tri and worst_val > 0:
            u, v, w = worst_tri
            # Strategy: Prioritize removing an edge incident to the highest degree vertex
            # within the worst triangle to break the triangle and reduce degree sum.
            
            # Get degrees of vertices in the triangle
            du, dv, dw = g.deg[u], g.deg[v], g.deg[w]
            
            # Sort edges in the triangle by the max degree of their endpoints (descending)
            # We want to remove the edge that involves the highest degree vertices
            tri_edges = [
                (u, v, max(du, dv)),
                (v, w, max(dv, dw)),
                (w, u, max(dw, du))
            ]
            tri_edges.sort(key=lambda x: x[2], reverse=True)
            
            # With high probability, remove one of these high-impact edges
            if random.random() < 0.7:
                rem_u, rem_v, _ = tri_edges[0]
            else:
                # Sometimes remove a random edge from the triangle or elsewhere
                # to explore the landscape
                if random.random() < 0.5:
                    rem_u, rem_v, _ = random.choice(tri_edges)
                else:
                    rem_u, rem_v = random.choice(list(g.edge_set))
        else:
            rem_u, rem_v = random.choice(list(g.edge_set))
        
        # 3. Choose edge to add
        rem_key = (min(rem_u, rem_v), max(rem_u, rem_v))
        
        # Heuristic for adding: 
        # If we just removed an edge to break a high-degree triangle, 
        # try to add an edge between low-degree vertices to keep degrees balanced,
        # or simply pick a random non-edge.
        
        # Let's try a slightly smarter addition: 
        # Pick a random non-edge, but if we have time, we could check if it creates 
        # a triangle with sum < current worst. For now, random is safer for SA.
        add_pair = g.get_random_non_edge(exclude=rem_key)
        
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
        if no_improve_count > 1000:
            # Reset temperature and restart from best
            g = BitGraph(n, list(best_edges))
            current_score = best_score
            T = 3.0
            no_improve_count = 0

    return list(best_edges)

def _polish(g, deadline):
    """
    Final polish: only accept moves that strictly improve the score.
    Uses a more exhaustive check of candidate swaps.
    """
    n = g.n
    current_score = g.get_score()
    best_score = current_score
    best_edges = set(g.edge_set)
    start_time = time.time()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Get worst triangle
        worst_tri, worst_val = g.get_worst_triangle()
        
        if worst_tri is None or worst_val == 0:
            break
        
        u, v, w = worst_tri
        
        # Candidate removals: edges in the worst triangle, plus some random high-degree edges
        candidates_to_remove = [(u, v), (v, w), (w, u)]
        
        # Also consider removing edges incident to the highest degree vertex in the triangle
        # that are not in the triangle, to lower its degree
        max_deg_vertex = max([u, v, w], key=lambda x: g.deg[x])
        
        # Add a few random edges incident to max_deg_vertex
        for _ in range(3):
            # Pick a random neighbor of max_deg_vertex
            neighbors = []
            adj_mask = g.adj[max_deg_vertex]
            while adj_mask:
                lsb = adj_mask & -adj_mask
                nb = lsb.bit_length() - 1
                neighbors.append(nb)
                adj_mask ^= lsb
            if neighbors:
                nb = random.choice(neighbors)
                candidates_to_remove.append((max_deg_vertex, nb))
        
        random.shuffle(candidates_to_remove)
        
        for rem_u, rem_v in candidates_to_remove:
            if not g.is_edge(rem_u, rem_v):
                continue
            rem_key = (min(rem_u, rem_v), max(rem_u, rem_v))
            
            # Try several candidate non-edges
            non_edges = g.get_all_non_edges()
            random.shuffle(non_edges)
            
            for add_u, add_v in non_edges[:20]: # Limit checks for speed
                if (add_u, add_v) == rem_key:
                    continue
                
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
                    break # Restart the outer loop with the new state
                else:
                    # Revert
                    g.remove_edge(add_u, add_v)
                    g.add_edge(rem_u, rem_v)
            
            if improved:
                break # Break out of removal candidates loop
        
    return list(best_edges)

def construct():
    """Construct a graph on 16 vertices with 65 edges minimizing max triangle degree sum."""
    start_time = time.time()
    # Evaluator kills at 150s. Use 145s for safety.
    deadline = start_time + 145
    
    n = 16
    target_edges = 65
    
    # Start with baseline
    initial_edges = _construct_baseline()
    
    # Run local search with most of the budget
    # Reserve 20s for polish
    search_deadline = start_time + (deadline - start_time) - 20
    best_edges = _local_search(n, initial_edges, search_deadline)
    
    # Final polish phase with remaining time
    polish_deadline = deadline
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
