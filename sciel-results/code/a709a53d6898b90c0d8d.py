import random
import time
from itertools import combinations

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    n = 20
    
    # Precompute all pairs for faster non-edge selection?
    # Or just maintain a set of edges.
    
    def has_k33_local(adj, u, v):
        """
        Check if adding edge (u, v) to graph 'adj' creates a K3,3.
        Assumes (u, v) is NOT in adj.
        Returns True if adding (u, v) creates a K3,3.
        """
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in adj:
            return False
            
        # Get common neighbors of u and v
        # nbrs_u = {x for (x, y) in adj if x == u or y == u}
        # This is slow if done from scratch.
        # We should maintain neighbor sets.
        pass

    # To make it fast, we must maintain neighbor sets.
    
    class Graph:
        def __init__(self):
            self.n = n
            self.adj = set()
            self.nbrs = [set() for _ in range(self.n)]
            
        def add_edge(self, u, v):
            if u == v:
                return False
            if u > v:
                u, v = v, u
            if (u, v) in self.adj:
                return False
            self.adj.add((u, v))
            self.nbrs[u].add(v)
            self.nbrs[v].add(u)
            return True

        def remove_edge(self, u, v):
            if u == v:
                return False
            if u > v:
                u, v = v, u
            if (u, v) not in self.adj:
                return False
            self.adj.remove((u, v))
            self.nbrs[u].discard(v)
            self.nbrs[v].discard(u)
            return True

        def is_valid_add(self, u, v):
            """
            Check if adding (u, v) keeps the graph K3,3-free.
            """
            if u == v:
                return False
            if u > v:
                u, v = v, u
            if (u, v) in self.adj:
                return False # Already exists, technically not a violation, but not an addition
            
            # Common neighbors of u and v
            C = self.nbrs[u] & self.nbrs[v]
            
            # We need to check if there exist a1, a2 in C and b1, b2 in C
            # such that {a1, a2} and {b1, b2} are disjoint and form a K2,2 in the graph.
            # The edges of K2,2 are (a1,b1), (a1,b2), (a2,b1), (a2,b2).
            # Note: a_i and b_j are in C, so they are neighbors of both u and v.
            # The edges (a_i, b_j) must exist in the current graph.
            
            # If |C| < 4, no K3,3 can be formed involving u and v as opposite parts.
            if len(C) < 4:
                return True
                
            # Iterate through combinations of 4 vertices from C
            # There are at most 18 vertices in C.
            # C(18, 4) = 3060.
            # For each combination, check if it contains a K2,2.
            # A K2,2 is a 4-cycle. In a set of 4 vertices, 
            # it forms a K2,2 if there are 4 edges forming a cycle?
            # No, K2,2 is a complete bipartite graph between two pairs.
            # For a set of 4 vertices {x, y, z, w}, it forms a K2,2 if 
            # we can partition into {x,y} and {z,w} such that x-z, x-w, y-z, y-w exist.
            # This is equivalent to checking if the subgraph induced by these 4 vertices
            # contains a K2,2.
            # There are 3 ways to partition 4 vertices into two pairs.
            # 1. {x,y} vs {z,w}
            # 2. {x,z} vs {y,w}
            # 3. {x,w} vs {y,z}
            
            for quad in combinations(C, 4):
                x, y, z, w = quad
                # Check partition 1: {x,y} and {z,w}
                if (x,z) in self.adj and (x,w) in self.adj and (y,z) in self.adj and (y,w) in self.adj:
                    return False
                # Check partition 2: {x,z} and {y,w}
                if (x,y) in self.adj and (x,w) in self.adj and (z,y) in self.adj and (z,w) in self.adj:
                    return False
                # Check partition 3: {x,w} and {y,z}
                if (x,y) in self.adj and (x,z) in self.adj and (w,y) in self.adj and (w,z) in self.adj:
                    return False
                    
            return True

        def find_violation(self):
            """Find any K3,3 in the current graph. Returns (u, v, w, x, y, z) or None."""
            # For each triple a, b, c
            for a, b, c in combinations(range(self.n), 3):
                common = self.nbrs[a] & self.nbrs[b] & self.nbrs[c]
                if len(common) >= 3:
                    B = list(common)[:3]
                    return (a, b, c, B[0], B[1], B[2])
            return None

    def repair(graph):
        """Remove edges to make graph K3,3-free."""
        # Limit iterations to avoid infinite loops or excessive time
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            viol = graph.find_violation()
            if viol is None:
                break
            A, B = viol[:3], viol[3:]
            # Remove a random edge from the K3,3
            # Edges are between A and B
            edges_to_remove = [(a, b) for a in A for b in B]
            # Normalize and check existence
            valid_edges = []
            for u, v in edges_to_remove:
                if u > v:
                    u, v = v, u
                if (u, v) in graph.adj:
                    valid_edges.append((u, v))
            if valid_edges:
                e = random.choice(valid_edges)
                graph.remove_edge(e[0], e[1])
            else:
                break

    def optimize(graph):
        """Try to add edges to maximize count."""
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Shuffle vertices to randomize order
            vertices = list(range(n))
            random.shuffle(vertices)
            for u in vertices:
                for v in vertices:
                    if u >= v:
                        continue
                    if (u, v) not in graph.adj:
                        if graph.is_valid_add(u, v):
                            graph.add_edge(u, v)
                            improved = True
            # If no improvement in a full pass, try a kick (remove an edge)
            if not improved and len(graph.adj) > 0:
                # Pick a random edge to remove
                e = random.choice(list(graph.adj))
                graph.remove_edge(e[0], e[1])
                # Try to add edges again
                # This is a simple kick. A better kick might be to remove an edge
                # that is "blocking" many additions.
                # For now, random kick.

    random.seed(42)
    
    best_graph = None
    best_count = -1
    
    # Multiple restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        # Initialize
        g = Graph()
        
        # Random initialization
        # Start with p=0.5
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.5:
                    g.add_edge(u, v)
        
        # Repair
        repair(g)
        
        # Optimize
        optimize(g)
        
        if len(g.adj) > best_count:
            best_count = len(g.adj)
            best_graph = g
            # Print progress for debugging (optional, but good for local runs)
            # print(f"Restart {r}: {best_count} edges")

    # Final result
    if best_graph is None:
        # Fallback to empty graph or a simple valid graph
        g = Graph()
        for u in range(n):
            for v in range(u+1, n):
                if g.is_valid_add(u, v):
                    g.add_edge(u, v)
        return [[u, v] for u, v in g.adj]
        
    return [[u, v] for u, v in best_graph.adj]
