import random
import sys

def construct():
    n = 15
    max_iter = 20000
    
    # Represent graph as a list of edge tuples or a bitset matrix
    # Using a set of frozensets for easy lookup, or just a list for return.
    # For fast validity check, use an adjacency matrix (list of lists of booleans) or bitmasks.
    
    def get_valid_edges(edges_set, adj_matrix, degrees):
        """
        Check if the graph defined by adj_matrix and degrees is valid.
        Returns True if valid.
        """
        # Optimization: If any two vertices have the same degree, they cannot be
        # endpoints of a 3-edge path in the same "layer".
        # Actually, the constraint is specific to paths s-u-v-t.
        
        for u in range(n):
            for v in range(u + 1, n):
                if not adj_matrix[u][v]:
                    continue
                
                # u and v are connected.
                # Neighbors of u (excluding v)
                # Neighbors of v (excluding u)
                # Any pair s in N(u)\{v}, t in N(v)\{u} with s!=t must have deg[s] != deg[t]
                
                # If N(u)\{v} and N(v)\{u} contain vertices with equal degrees,
                # specifically if there exists s in N(u)\{v} and t in N(v)\{u} 
                # such that deg[s] == deg[t] and s != t, then it's invalid.
                
                # Note: If s == t, it's not a simple path s-u-v-t (s must be distinct from t).
                # The problem states "simple three-edge path s-u-v-t". 
                # Simple means all vertices distinct. So s, u, v, t are all distinct.
                
                # So we check all s in adj[u] where s!=v and s!=t (implicitly t!=s)
                # and t in adj[v] where t!=u and t!=s.
                
                # Efficient check:
                # For each degree value d, let S_d be the set of vertices with degree d.
                # The constraint is violated if there exists u-v edge, and
                # | (N(u)\{v} intersect S_d) | >= 1 AND | (N(v)\{u} intersect S_d) | >= 1
                # AND they don't share the same vertex s (which is excluded by s!=t).
                # Actually, if s != t is required, we just need to ensure that
                # the sets N(u)\{v} and N(v)\{u} do not contain vertices of the same degree
                # that are distinct.
                
                # Simpler: Just iterate. N is small (15).
                for s in range(n):
                    if s == v or s == u:
                        continue
                    if not adj_matrix[u][s]:
                        continue
                    deg_s = degrees[s]
                    for t in range(n):
                        if t == s or t == u or t == v:
                            continue
                        if not adj_matrix[v][t]:
                            continue
                        if degrees[t] == deg_s:
                            return False
        return True

    # Initialization: Start with K_7,8 (56 edges) as a known valid base?
    # Or start random and climb.
    # Let's start with the known good one and try to add edges.
    
    # Build K_7,8
    adj = [[False] * n for _ in range(n)]
    edges_set = set()
    for i in range(7):
        for j in range(7, 15):
            adj[i][j] = True
            adj[j][i] = True
            edges_set.add(frozenset([i, j]))
    
    # Calculate initial degrees
    degrees = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                degrees[i] += 1
                
    current_best = len(edges_set)
    best_adj = [row[:] for row in adj]
    best_edges = list(edges_set)
    
    # Local Search: Try to add random edges
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
    
    # Shuffle to randomize order
    random.seed(42)
    
    # We will perform a greedy search: try to add an edge if it keeps the graph valid.
    # If we get stuck, we might need to remove some edges to allow adding others, 
    # but simple hill climbing on addition first is a good start.
    
    # To be more robust, use Simulated Annealing or simple restarts.
    # Given 60s limit, we can afford many iterations.
    
    for iteration in range(10000):
        # Try adding a random edge
        if not all_possible_edges:
            break
        idx = random.randint(0, len(all_possible_edges) - 1)
        u, v = all_possible_edges[idx]
        
        # Tentatively add
        adj[u][v] = True
        adj[v][u] = True
        degrees[u] += 1
        degrees[v] += 1
        
        if get_valid_edges(edges_set, adj, degrees):
            edges_set.add(frozenset([u, v]))
            all_possible_edges.pop(idx)
            current_best = len(edges_set)
            # Update best if improved
            if current_best > len(best_edges):
                best_edges = list(edges_set)
                best_adj = [row[:] for row in adj]
        else:
            # Revert
            adj[u][v] = False
            adj[v][u] = False
            degrees[u] -= 1
            degrees[v] -= 1
            
    # Return the best found edges
    # Ensure types are correct: list of lists of ints
    final_edges = []
    for e in best_edges:
        u, v = tuple(e)
        final_edges.append([u, v])
        
    return final_edges
