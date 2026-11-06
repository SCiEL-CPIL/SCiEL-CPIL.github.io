import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Build the graph
    # Candidates are numbers in [1, 255] with gcd(x, 255) > 1
    candidates = [x for x in range(1, 256) if math.gcd(x, 255) > 1]
    n = len(candidates)
    
    # Map value to index for quick lookup
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Build adjacency matrix as list of bitmasks
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Index of 255
    idx_255 = val_to_idx[255]
    
    best_clique_size = 1
    best_clique_mask = 1 << idx_255
    
    def popcount(x):
        return bin(x).count('1')
    
    # Tomita's Coloring
    # We need to color the subgraph induced by P such that adjacent nodes have different colors.
    # The color assigned to a node v is an upper bound on the clique size starting from v.
    # We process nodes in order.
    
    def color_sort(P):
        # Returns list of (node, color) sorted by color descending
        # Greedy coloring:
        # Unordered list of nodes in P
        nodes = []
        temp = P
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            temp ^= lsb
            nodes.append(v)
        
        # Simple greedy coloring
        # color[v] is the color index
        color = {}
        # We want to assign colors 1, 2, ...
        # For each node, assign the smallest color not used by its neighbors in the processed set?
        # Actually, standard Tomita coloring:
        # 1. Start with empty set of colored nodes.
        # 2. Iterate through nodes.
        # 3. Assign color to node.
        # To get a good upper bound, we usually color in a way that maximizes the color values for the "end" of the sequence.
        
        # A simpler effective heuristic:
        # Sort nodes by degree in P (descending) might help?
        # Let's stick to a basic sequential coloring.
        
        # We will assign colors 1..k.
        # For each node v in nodes:
        #   used_colors = {color[u] for u in neighbors_of_v_in_P if u in color}
        #   color[v] = smallest positive int not in used_colors
        # This gives a valid coloring.
        # The property we need: If we process nodes in order v1, v2... vk,
        # and color[v_i] = c, then the max clique size in the subgraph induced by {v_i...vk} is at most c?
        # No, the property is: If we color the graph, the max clique size is at most the number of colors.
        # In the recursive step, we consider vertices in order of decreasing color.
        # If we are at vertex v with color c, and we have already found a clique of size best,
        # and |R| + c <= best, we can prune.
        
        # Let's implement a standard greedy coloring.
        # To make the pruning effective, we want high colors for vertices that are likely to be in large cliques?
        # Actually, the color is an upper bound.
        
        # Let's try a specific ordering:
        # We will return a list of nodes sorted by their assigned color, descending.
        
        # Greedy coloring algorithm
        # 1. Order vertices (e.g., by degree in P, descending)
        # 2. Assign colors
        
        # Degrees in P
        degrees = []
        for v in nodes:
            deg = popcount(P & adj[v])
            degrees.append((deg, v))
        degrees.sort(reverse=True)
        
        ordered_nodes = [v for _, v in degrees]
        
        color_map = {}
        # For each node, check neighbors that have already been colored
        # We want to assign the smallest possible color?
        # If we assign smallest possible, the max color is the chromatic number (approx).
        # The bound is: max clique <= max color.
        
        for v in ordered_nodes:
            # Get colors of neighbors in P that are already colored
            neighbors_in_p = P & adj[v]
            used_colors = set()
            temp = neighbors_in_p
            while temp:
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                temp ^= lsb
                if u in color_map:
                    used_colors.add(color_map[u])
            
            # Find smallest available color
            c = 1
            while c in used_colors:
                c += 1
            color_map[v] = c
        
        # Create list of (color, node)
        colored_nodes = [(color_map[v], v) for v in nodes]
        # Sort by color descending
        colored_nodes.sort(reverse=True)
        
        return colored_nodes

    def bron_kerbosch(R, P, X):
        nonlocal best_clique_size, best_clique_mask
        if time.time() > deadline:
            return
        
        if P == 0 and X == 0:
            sz = popcount(R)
            if sz > best_clique_size:
                best_clique_size = sz
                best_clique_mask = R
            return
        
        # Get colored nodes
        colored_nodes = color_sort(P)
        
        for color, v in colored_nodes:
            if time.time() > deadline:
                return
            
            # Pruning condition
            if popcount(R) + color <= best_clique_size:
                break  # Since colors are descending, all subsequent nodes have <= color
            
            # Recurse
            new_R = R | (1 << v)
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            # Remove v from P for subsequent iterations?
            # In standard BK, we iterate over P.
            # Here we iterate over the sorted list.
            # We must ensure we don't process the same node again.
            # The standard BK loop removes v from P after processing.
            # Since we are using a pre-sorted list, we just continue to the next item in the list.
            # However, the definition of P changes? No, P is fixed for this level.
            # The recursive call uses new_P.
            # We just need to make sure we don't recurse on v again in the same loop.
            # The loop structure handles this by moving to the next item in colored_nodes.
            
            bron_kerbosch(new_R, new_P, new_X)
            
            # Remove v from P?
            # In the standard algorithm:
            # for v in P:
            #   BK(R+{v}, P \ {v} \ N(v), X \ N(v)) -- Wait, standard is:
            #   BK(R+{v}, P & N(v), X & N(v))
            #   P = P \ {v}
            #   X = X \ {v}
            
            # My current loop iterates over a static list `colored_nodes`.
            # I need to simulate the removal.
            # Actually, the standard BK with pivot iterates over P \ N(pivot).
            # Tomita's algorithm usually iterates over P in order of color.
            # The condition `if |R| + color[v] <= best: break` works if we process in decreasing color.
            # We don't strictly need to remove v from P for the logic of the *next* iteration in the *same* level,
            # because the recursive call handles the subgraph.
            # However, we must ensure we don't generate the same clique multiple times.
            # The X set handles the "already processed" part.
            # In the standard BK:
            #   for v in P:
            #      ...
            #      P.remove(v)
            #      X.add(v)
            #
            # In my implementation, I am not updating P or X inside the loop.
            # This is incorrect for standard BK.
            # I need to maintain the state of P and X as I iterate.
            
            # Let's refactor to properly update P and X.
            # But `colored_nodes` is a static list.
            # I can just filter it.
            pass # Placeholder, will rewrite function

    # Refactored Bron-Kerbosch with Tomita Coloring
    def bron_kerbosch_tomita(R, P, X):
        nonlocal best_clique_size, best_clique_mask
        if time.time() > deadline:
            return
        
        if P == 0 and X == 0:
            sz = popcount(R)
            if sz > best_clique_size:
                best_clique_size = sz
                best_clique_mask = R
            return
        
        # Get colored nodes for current P
        # We need to color the subgraph induced by P.
        colored_nodes = color_sort(P)
        
        # We will iterate through these nodes.
        # We must maintain P and X as we go.
        # However, the colors were computed for the *initial* P.
        # As we remove nodes from P, the coloring might become invalid (colors might be too low? No, removing nodes can only decrease clique size, so upper bound holds? 
        # Actually, if we remove a node, the color of remaining nodes might be over-estimates, which is safe for pruning (we might prune less, but not incorrectly).
        # Wait, if we remove a node, the max clique in the remaining P is <= max clique in original P.
        # The color assigned was an upper bound for the original P.
        # Is it an upper bound for the subset? Yes.
        
        # We need to iterate and update P, X.
        # Since `colored_nodes` is a list of (color, v), we can just loop through it.
        # But we must skip nodes that are no longer in P (if we were modifying P).
        # In standard BK, we remove v from P after processing.
        
        # Let's use a mutable copy of P and X?
        # No, we can just use the list and a set of "processed" nodes?
        # Or better, just follow the standard algorithm structure but with the coloring heuristic.
        
        # Standard Tomita MCQ:
        # 1. Color P.
        # 2. For each v in P (ordered by color desc):
        #    if |R| + color[v] <= best: return
        #    recurse(R+{v}, P \ {v} \ N(v) ??? No, P & N(v), X & N(v))
        #    P = P \ {v}
        #    X = X \ {v}
        
        # The tricky part: The colors were computed for the full P.
        # When we remove v, the colors of other nodes are still valid upper bounds.
        
        # Implementation detail:
        # We need to iterate over the nodes in `colored_nodes`.
        # We need to know if a node is still in P.
        # We can just check `if (1 << v) & P`.
        
        for color, v in colored_nodes:
            if time.time() > deadline:
                return
            
            # Check if v is still in P
            if not (P & (1 << v)):
                continue
                
            # Pruning
            if popcount(R) + color <= best_clique_size:
                break  # Since sorted desc, we can stop
            
            # Recurse
            new_R = R | (1 << v)
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            bron_kerbosch_tomita(new_R, new_P, new_X)
            
            # Update P and X for next iteration
            P = P & ~(1 << v)
            X = X | (1 << v)

    # Start with 255 forced
    R = 1 << idx_255
    P = adj[idx_255]
    X = 0
    
    bron_kerbosch_tomita(R, P, X)
    
    # Convert back
    result_set = set()
    for i in range(n):
        if (best_clique_mask >> i) & 1:
            result_set.add(candidates[i])
            
    if 255 not in result_set:
        result_set.add(255)
        
    # Verify
    res_list = list(result_set)
    for i in range(len(res_list)):
        for j in range(i + 1, len(res_list)):
            if math.gcd(res_list[i], res_list[j]) == 1:
                return list(range(3, 256, 3))
                
    if 255 not in res_list:
        return list(range(3, 256, 3))
        
    return res_list

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
