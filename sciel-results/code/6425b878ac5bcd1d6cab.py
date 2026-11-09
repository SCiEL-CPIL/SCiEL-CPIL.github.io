import random
import time

def construct():
    """
    Construct a C4-free graph on 20 vertices to satisfy the rainbow 4-cycle constraint vacuously.
    Maximizes edge count by using a greedy random algorithm.
    """
    n = 20
    best_edges = []
    start_time = time.time()
    timeout = 120  # seconds
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Run greedy multiple times to find a denser C4-free subgraph
    for attempt in range(50):
        if time.time() - start_time > timeout - 5:
            break
            
        random.shuffle(all_edges)
        edges = set()
        
        for u, v in all_edges:
            # Check if adding (u, v) creates a C4
            # A C4 is created if there exists a vertex w such that (u, w) and (v, w) are in edges
            # This forms the cycle u-w-v-... wait.
            # Cycle: u - v - w - x - u ? No.
            # If we add (u, v), a new 4-cycle is formed if there is a path of length 2 between u and v.
            # Path u-w-v implies edges (u, w) and (w, v).
            # The cycle would be u-v-w-... no, that's a triangle if u-w and w-v exist?
            # Let's trace:
            # Existing edges: (u, w) and (v, w).
            # New edge: (u, v).
            # Cycle: u - v - w - u. This is a triangle (3-cycle), not a 4-cycle.
            
            # Wait, the condition for a 4-cycle involving new edge (u, v) is:
            # Is there a path of length 3 between u and v? No, that would make a 4-cycle u-a-b-v-u.
            # Or is there a path of length 1? No, that's a double edge.
            # Let's re-evaluate.
            # A 4-cycle is $v_1 - v_2 - v_3 - v_4 - v_1$.
            # If we add edge $(u, v)$, it can be part of a 4-cycle if:
            # 1. There is a path of length 3 from u to v in the existing graph.
            #    i.e., $\exists a, b$ such that $(u, a), (a, b), (b, v)$ are edges.
            # 2. OR, is it possible that the new edge completes a 4-cycle in another way?
            #    No, the new edge is one side of the cycle. The other 3 sides must exist.
            #    So we need a path of length 3 between u and v.
            
            # However, checking for a path of length 3 is more expensive than length 2.
            # Let's check the definition of C4-free.
            # A graph is C4-free if it does not contain a cycle of length 4.
            # When adding edge (u, v), we must ensure no 4-cycle is formed.
            # Any 4-cycle using (u, v) must be of the form u-v-a-b-u.
            # This requires edges (v, a), (a, b), (b, u) to exist.
            # So we need to check if there is a path of length 3 from u to v.
            
            # Is there a faster way?
            # For small n=20, checking paths of length 3 is fine.
            # Neighbors of u: N(u). Neighbors of v: N(v).
            # We need to check if there is a, b such that a in N(v), b in N(u), and (a, b) is an edge.
            
            is_c4 = False
            # Optimization: Iterate over common neighbors? No, length 3 path.
            # Let's iterate over all pairs (a, b) where a is neighbor of v and b is neighbor of u?
            # Or simpler:
            # For each a in N(v):
            #   For each b in N(u):
            #     If (a, b) is an edge, then u-b-a-v-u is a 4-cycle.
            
            # Let's get neighbors
            n_u = [x for x in range(n) if (min(u, x), max(u, x)) in edges]
            n_v = [x for x in range(n) if (min(v, x), max(v, x)) in edges]
            
            for a in n_v:
                for b in n_u:
                    if a == b:
                        continue # This would be a triangle u-v-a-u? No, u-v and v-a and a-u. Triangle.
                    # Check if (a, b) is an edge
                    edge_ab = (min(a, b), max(a, b))
                    if edge_ab in edges:
                        is_c4 = True
                        break
                if is_c4:
                    break
            
            if not is_c4:
                edges.add((u, v))
        
        # Convert to required format
        # Colors can be arbitrary since there are no 4-cycles
        current_edges = [[u, v, 0] for u, v in edges]
        
        if len(current_edges) > len(best_edges):
            best_edges = current_edges
            
    return best_edges
