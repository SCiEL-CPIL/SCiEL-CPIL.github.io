import numpy as np
import random
import time
import sys

N = 31
E = 91
DEADLINE_SECONDS = 118.0  # safety margin under the 150 s limit

# --- Polarity Graph Construction ---

def get_points_q5():
    """
    Generate the 31 points of PG(2,5) as representative vectors in F_5^3.
    """
    points = []
    # We need to pick one representative from each 1D subspace.
    # A point is [x:y:z]. We can fix the first non-zero coordinate to 1.
    # Cases:
    # 1. x != 0: x=1, y in 0..4, z in 0..4 -> 25 points
    # 2. x == 0, y != 0: y=1, z in 0..4 -> 5 points
    # 3. x == 0, y == 0, z != 0: z=1 -> 1 point
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    for z in range(5):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    return points

def build_polarity_graph():
    """
    Construct the polarity graph of PG(2,5).
    Returns adjacency list and edge list.
    """
    points = get_points_q5()
    # Map point tuple to index
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Polarity matrix M for PG(2,5).
    # We use a symmetric matrix. 
    # x^T M y = 0 defines adjacency.
    # A standard polarity for odd q is given by a matrix with 0 on diagonal?
    # Let's try M = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    # This corresponds to x1*y2 + x2*y1 + x3*y3 = 0.
    M = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 1]], dtype=int)
    
    adj = np.zeros((N, N), dtype=int)
    edges = []
    
    for i in range(N):
        for j in range(i + 1, N):
            pi = np.array(points[i])
            pj = np.array(points[j])
            # Check if pi is on polar line of pj: pi^T M pj = 0
            val = (pi @ M @ pj) % 5
            if val == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append((i, j))
                
    # Check if we have 90 edges. If not, we might need a different M.
    # For M = [[0,1,0],[1,0,0],[0,0,1]], the number of absolute points (loops) 
    # are solutions to x1*y2 + x2*y1 + x3^2 = 0 with x=y.
    # 2 x1 x2 + x3^2 = 0.
    # If this yields 6 absolute points, we have 90 edges.
    # If the edge count is not 90, we can try another M, e.g., [[0,1,0],[1,0,1],[0,1,0]]?
    # But 90 is the expected max. Let's assume this M works or is close.
    # If it's not 90, we can just use the graph as is and add edges until we have 91.
    # However, the problem states ex(31, C4)=90, so a 90-edge C4-free graph exists.
    
    return adj, edges

def count_c4_bitmask(adj_masks):
    """
    Count C4s using bitmasks.
    C4 = 1/2 * sum_{u < v} C(popcount(adj[u] & adj[v]), 2)
    """
    total = 0
    for u in range(N):
        for v in range(u + 1, N):
            common = adj_masks[u] & adj_masks[v]
            k = bin(common).count('1')
            total += k * (k - 1) // 2
    return total // 2

def construct():
    start_time = time.time()
    
    # 1. Build the base polarity graph
    adj_base, edges_base = build_polarity_graph()
    
    # If the base graph doesn't have exactly 90 edges, we need to adjust.
    # But typically it does. Let's assume it has E_base edges.
    # We need E=91 edges.
    # If E_base < 91, we add random non-edges.
    # If E_base == 90, we add the best non-edge.
    # If E_base > 91, we remove random edges.
    
    # For robustness, let's start with the base graph and ensure we have 91 edges.
    current_edges = edges_base[:]
    current_edge_set = set(current_edges)
    
    # Build adjacency masks for the base graph
    adj_masks = [0] * N
    for u, v in current_edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    
    # If we have fewer than 91 edges, add more
    # Find non-edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    non_edges = [e for e in all_edges if e not in current_edge_set]
    
    # Greedily add edges to reach 91
    while len(current_edges) < E:
        # Pick a random non-edge
        # Or pick the one with min C4 increase? 
        # For simplicity, pick random from non_edges
        if not non_edges:
            break
        idx = random.randint(0, len(non_edges) - 1)
        u, v = non_edges[idx]
        non_edges.pop(idx)
        
        current_edges.append((u, v))
        current_edge_set.add((u, v))
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    # If we have more than 91 edges (unlikely if base is 90), remove random ones
    while len(current_edges) > E:
        idx = random.randint(0, len(current_edges) - 1)
        u, v = current_edges[idx]
        current_edges.pop(idx)
        current_edge_set.remove((u, v))
        adj_masks[u] &= ~(1 << v)
        adj_masks[v] &= ~(1 << u)
        
    # Now we have exactly 91 edges.
    # Calculate initial C4 count
    current_c4 = count_c4_bitmask(adj_masks)
    
    best_c4 = current_c4
    best_edges = current_edges[:]
    best_masks = adj_masks[:]
    
    # 2. Local Search with Bitmasks
    # Simulated Annealing
    T = 10.0
    cooling = 0.98
    
    # Precompute list of non-edges for faster selection?
    # It changes, so we might just sample.
    
    deadline = start_time + DEADLINE_SECONDS
    
    while time.time() < deadline - 2:
        # Pick a random edge to remove
        remove_idx = random.randint(0, E - 1)
        u1, v1 = current_edges[remove_idx]
        
        # Pick a random non-edge to add
        # Sample a random pair
        valid = False
        u2, v2 = 0, 0
        for _ in range(20):
            u2 = random.randint(0, N - 1)
            v2 = random.randint(0, N - 1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            if (u2, v2) not in current_edge_set:
                valid = True
                break
        
        if not valid:
            continue
            
        # Perform swap
        # Remove (u1, v1)
        adj_masks[u1] &= ~(1 << v1)
        adj_masks[v1] &= ~(1 << u1)
        current_edge_set.remove((u1, v1))
        
        # Add (u2, v2)
        adj_masks[u2] |= (1 << v2)
        adj_masks[v2] |= (1 << u1) # Typo check: should be (1 << u2)
        adj_masks[v2] |= (1 << u2)
        
        current_edges[remove_idx] = (u2, v2)
        current_edge_set.add((u2, v2))
        
        new_c4 = count_c4_bitmask(adj_masks)
        
        delta = new_c4 - current_c4
        
        # Accept if better or with prob exp(-delta/T)
        if delta <= 0 or (T > 0.01 and random.random() < np.exp(-delta / T)):
            current_c4 = new_c4
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = current_edges[:]
                best_masks = adj_masks[:]
            T *= cooling
        else:
            # Revert
            adj_masks[u2] &= ~(1 << v2)
            adj_masks[v2] &= ~(1 << u2)
            current_edge_set.remove((u2, v2))
            
            adj_masks[u1] |= (1 << v1)
            adj_masks[v1] |= (1 << u1)
            current_edge_set.add((u1, v1))
            
            current_edges[remove_idx] = (u1, v1)
            T *= cooling
            
        if best_c4 <= 1:
            break
            
    return best_edges

# Fix the typo in the swap logic in the thought process above:
# In the code block below, I will ensure the bitmask update is correct.

def construct():
    start_time = time.time()
    
    # 1. Build the base polarity graph
    adj_base, edges_base = build_polarity_graph()
    
    current_edges = edges_base[:]
    current_edge_set = set(current_edges)
    
    # Build adjacency masks for the base graph
    adj_masks = [0] * N
    for u, v in current_edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    
    # Ensure we have exactly 91 edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    non_edges = [e for e in all_edges if e not in current_edge_set]
    
    while len(current_edges) < E:
        if not non_edges:
            break
        idx = random.randint(0, len(non_edges) - 1)
        u, v = non_edges[idx]
        non_edges.pop(idx)
        
        current_edges.append((u, v))
        current_edge_set.add((u, v))
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    while len(current_edges) > E:
        idx = random.randint(0, len(current_edges) - 1)
        u, v = current_edges[idx]
        current_edges.pop(idx)
        current_edge_set.remove((u, v))
        adj_masks[u] &= ~(1 << v)
        adj_masks[v] &= ~(1 << u)
        
    current_c4 = count_c4_bitmask(adj_masks)
    
    best_c4 = current_c4
    best_edges = current_edges[:]
    
    T = 10.0
    cooling = 0.98
    deadline = start_time + DEADLINE_SECONDS
    
    while time.time() < deadline - 2:
        remove_idx = random.randint(0, E - 1)
        u1, v1 = current_edges[remove_idx]
        
        valid = False
        u2, v2 = 0, 0
        for _ in range(20):
            u2 = random.randint(0, N - 1)
            v2 = random.randint(0, N - 1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            if (u2, v2) not in current_edge_set:
                valid = True
                break
        
        if not valid:
            continue
            
        # Remove (u1, v1)
        adj_masks[u1] &= ~(1 << v1)
        adj_masks[v1] &= ~(1 << u1)
        current_edge_set.remove((u1, v1))
        
        # Add (u2, v2)
        adj_masks[u2] |= (1 << v2)
        adj_masks[v2] |= (1 << u2)
        
        current_edges[remove_idx] = (u2, v2)
        current_edge_set.add((u2, v2))
        
        new_c4 = count_c4_bitmask(adj_masks)
        
        delta = new_c4 - current_c4
        
        if delta <= 0 or (T > 0.01 and random.random() < np.exp(-delta / T)):
            current_c4 = new_c4
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = current_edges[:]
            T *= cooling
        else:
            # Revert
            adj_masks[u2] &= ~(1 << v2)
            adj_masks[v2] &= ~(1 << u2)
            current_edge_set.remove((u2, v2))
            
            adj_masks[u1] |= (1 << v1)
            adj_masks[v1] |= (1 << u1)
            current_edge_set.add((u1, v1))
            
            current_edges[remove_idx] = (u1, v1)
            T *= cooling
            
        if best_c4 <= 1:
            break
            
    return best_edges
