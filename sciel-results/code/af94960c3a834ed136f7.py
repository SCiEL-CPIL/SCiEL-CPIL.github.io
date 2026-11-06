import random
import time
import sys

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Generate 1-factorization of K_20
    # Vertices 0..19. Let 19 be infinity.
    # Factors 0..18
    # For factor k in 0..18:
    #   Edge (19, k)
    #   Edges ((k+i) % 19, (k-i) % 19) for i in 1..9
    
    edges = []
    edge_colors = {}
    adj = [[False]*n for _ in range(n)]
    color_mat = [[-1]*n for _ in range(n)]
    
    edge_list = [] # List of (u, v)
    
    for k in range(19):
        # Edge (19, k)
        u, v = 19, k
        edge_list.append((u, v))
        edge_colors[(u, v)] = k
        edge_colors[(v, u)] = k
        adj[u][v] = True
        adj[v][u] = True
        color_mat[u][v] = k
        color_mat[v][u] = k
        
        for i in range(1, 10):
            u = (k + i) % 19
            v = (k - i) % 19
            edge_list.append((u, v))
            edge_colors[(u, v)] = k
            edge_colors[(v, u)] = k
            adj[u][v] = True
            adj[v][u] = True
            color_mat[u][v] = k
            color_mat[v][u] = k
            
    # Total 19 * 10 = 190 edges
    
    # 2. Calculate initial bad cycle counts for each edge
    # bad_count[e] = number of bad 4-cycles containing e
    # A bad 4-cycle consists of 4 edges.
    # We iterate over colors. For each color, we have 10 edges.
    # For each pair of edges in that color, we check if they form bad 4-cycles.
    
    bad_count = {e: 0 for e in edge_list}
    
    # To update counts efficiently, we need to know which edges are in bad cycles with which other edges.
    # conflict[e1] = set of e2 such that e1, e2 are opposite in a bad 4-cycle.
    # Actually, if e1 and e2 are opposite in a bad 4-cycle, the other 2 edges must also be present.
    # Since we start with K_20, all edges are present.
    # So for any pair of same-color edges, they form 2 bad 4-cycles (provided the other edges exist).
    
    # Let's just recompute or maintain a structure.
    # Given n=20, 190 edges.
    # Pairs per color: 45. Total pairs 19*45 = 855.
    # Each pair gives 2 bad cycles. Total 1710 bad cycles.
    # Each bad cycle has 4 edges.
    # Total edge-cycle incidences: 1710 * 4 = 6840.
    # Average count per edge: 6840 / 190 = 36?
    # Wait, earlier I said 18.
    # Edge e of color c. 9 other edges of color c.
    # Each other edge f forms 2 bad cycles with e.
    # So 9 * 2 = 18 bad cycles containing e.
    # Why 36?
    # Because each bad cycle has 2 edges of color c and 2 of other colors.
    # The count "bad_count[e]" counts the number of bad cycles containing e.
    # So yes, 18.
    
    # Let's initialize bad_count to 18 for all edges.
    for e in edge_list:
        bad_count[e] = 18
        
    # We need a way to update bad_count when an edge is removed.
    # If we remove edge e, we must decrement bad_count for all edges that were in bad cycles with e.
    # Which edges are those?
    # Let e = (u,v) with color c.
    # The other edges in bad cycles with e are:
    # 1. The 9 other edges of color c. (Opposite to e in some 4-cycle)
    # 2. The edges that form the other 2 sides of those 4-cycles.
    
    # This is complex to track dynamically without a full conflict graph.
    # Alternative: Just recompute bad_count from scratch?
    # 1710 cycles. Checking existence of 4 edges is O(1).
    # 1710 checks per removal.
    # If we remove ~100 edges, 1710 * 100 = 171,000 operations. Very fast.
    
    def find_bad_cycle():
        """Returns a bad 4-cycle (u, v, w, x) or None."""
        # Iterate colors
        # Group edges by color
        edges_by_color = {c: [] for c in range(19)}
        for u, v in edge_list:
            if adj[u][v]:
                c = color_mat[u][v]
                edges_by_color[c].append((u, v))
                
        for c in range(19):
            es = edges_by_color[c]
            # For each pair
            for i in range(len(es)):
                u1, v1 = es[i]
                if not adj[u1][v1]: continue
                for j in range(i+1, len(es)):
                    u2, v2 = es[j]
                    if not adj[u2][v2]: continue
                    
                    # Check cycle 1: u1-v1-v2-u2-u1
                    # Edges: (u1,v1), (v1,v2), (v2,u2), (u2,u1)
                    # Opposites: (u1,v1) & (v2,u2) [Color c]
                    # Need (v1,v2) and (u2,u1) to exist.
                    if adj[v1][v2] and adj[u2][u1]:
                        return (u1, v1, v2, u2)
                    
                    # Check cycle 2: u1-v1-u2-v2-u1
                    # Edges: (u1,v1), (v1,u2), (u2,v2), (v2,u1)
                    # Opposites: (u1,v1) & (u2,v2) [Color c]
                    # Need (v1,u2) and (v2,u1) to exist.
                    if adj[v1][u2] and adj[v2][u1]:
                        return (u1, v1, u2, v2)
                        
        return None

    removed_edges = set()
    
    while time.time() < deadline - 2:
        cycle = find_bad_cycle()
        if cycle is None:
            break
            
        u, v, w, x = cycle
        # Edges in cycle: (u,v), (v,w), (w,x), (x,u)
        # Note: The cycle returned is a sequence of vertices.
        # Let's identify the 4 edges.
        e1 = (u, v)
        e2 = (v, w)
        e3 = (w, x)
        e4 = (x, u)
        
        # Ensure edges are in edge_list format (min, max) for lookup?
        # My edge_list has (u,v) as generated.
        # adj is symmetric.
        
        # Count bad cycles for each of these 4 edges.
        # Since we are removing one, we should pick the one with highest "impact".
        # Impact = number of other bad cycles it is in.
        # We can approximate by current bad_count.
        
        # Update bad_count? 
        # If we just recompute find_bad_cycle, we don't strictly need accurate bad_count 
        # if we pick randomly or by static priority.
        # But picking the one in most cycles is better.
        
        # Let's just pick the edge with the highest degree in the current graph?
        # Or random.
        
        # Let's use bad_count.
        # But bad_count is stale.
        # Let's just pick the edge that is in the most "potential" conflicts.
        # In K_20, all are equal.
        # As we remove edges, some become less critical.
        
        # Simple heuristic: Remove the edge with the highest degree in the remaining graph.
        deg_e1 = sum(1 for k in range(n) if adj[u][k])
        deg_e2 = sum(1 for k in range(n) if adj[v][k])
        deg_e3 = sum(1 for k in range(n) if adj[w][k])
        deg_e4 = sum(1 for k in range(n) if adj[x][k])
        
        # Actually, removing an edge reduces the degree of its endpoints.
        # Let's just pick one randomly from the 4.
        # Or pick the one with the highest color frequency? No.
        
        # Let's pick the edge that appears in the most bad cycles.
        # We can do a quick count for these 4 edges.
        
        def count_bad_for_edge(e):
            u, v = e
            c = color_mat[u][v]
            count = 0
            # Find all other edges of color c
            for u2, v2 in edge_list:
                if (u2, v2) == (u, v) or (u2, v2) == (v, u): continue
                if not adj[u2][v2]: continue
                if color_mat[u2][v2] != c: continue
                
                # Check if they form a bad cycle
                # Cycle 1: u-v-v2-u2-u
                if adj[v][v2] and adj[u2][u]:
                    count += 1
                # Cycle 2: u-v-u2-v2-u
                if adj[v][u2] and adj[v2][u]:
                    count += 1
            return count

        c1 = count_bad_for_edge(e1)
        c2 = count_bad_for_edge(e2)
        c3 = count_bad_for_edge(e3)
        c4 = count_bad_for_edge(e4)
        
        max_count = max(c1, c2, c3, c4)
        candidates = []
        if c1 == max_count: candidates.append(e1)
        if c2 == max_count: candidates.append(e2)
        if c3 == max_count: candidates.append(e3)
        if c4 == max_count: candidates.append(e4)
        
        e_to_remove = random.choice(candidates)
        
        # Remove edge
        u, v = e_to_remove
        adj[u][v] = False
        adj[v][u] = False
        removed_edges.add(e_to_remove)
        
    # Construct final edge list
    final_edges = []
    for u, v in edge_list:
        if adj[u][v]:
            final_edges.append([u, v, color_mat[u][v]])
            
    return final_edges

# Test locally
if __name__ == "__main__":
    edges = construct()
    print(len(edges))
