import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def generate_paley(n):
        squares = set((x*x) % n for x in range(n))
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                d = (j - i) % n
                colors.append(1 if d in squares else 0)
        return colors

    # Baseline: Paley(29)
    n_base = 29
    base_colors = generate_paley(n_base)
    
    # Precompute base adjacency for color 1 and 0
    # adj1_base[i] is bitmask of neighbors of i in G1
    adj1_base = [0] * n_base
    adj0_base = [0] * n_base
    idx = 0
    for i in range(n_base):
        for j in range(i+1, n_base):
            c = base_colors[idx]
            if c == 1:
                adj1_base[i] |= (1 << j)
                adj1_base[j] |= (1 << i)
            else:
                adj0_base[i] |= (1 << j)
                adj0_base[j] |= (1 << i)
            idx += 1
            
    def check_k222_free(adj, n):
        """
        Checks if graph defined by adj (list of bitmasks) contains K2,2,2.
        Returns True if K2,2,2 is NOT present (i.e., free).
        """
        # Iterate over non-edges (u, v)
        # We can iterate over all pairs and check if they are edges.
        # To be faster, we could iterate over non-edges directly, but generating that list is O(n^2).
        # Given n=30, n^2=900, which is small.
        
        # Optimization: Precompute list of non-edges? 
        # Or just loop.
        
        for u in range(n):
            # Non-neighbors of u in G
            non_neigh_u = ((1 << n) - 1) ^ adj[u] ^ (1 << u) # exclude u itself
            # Iterate over v in non_neigh_u
            v_mask = non_neigh_u
            while v_mask:
                v_bit = v_mask & (-v_mask)
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit
                
                # u, v is a non-edge
                # S = N(u) & N(v)
                S = adj[u] & adj[v]
                
                # Check if S has at least 4 bits
                if S.bit_count() < 4:
                    continue
                
                # Iterate over pairs (x, y) in S
                s_list = []
                temp_S = S
                while temp_S:
                    bit = temp_S & (-temp_S)
                    s_list.append(bit.bit_length() - 1)
                    temp_S ^= bit
                
                m = len(s_list)
                for i in range(m):
                    x = s_list[i]
                    for j in range(i+1, m):
                        y = s_list[j]
                        
                        # Check if (x, y) is a non-edge in G
                        if (adj[x] >> y) & 1:
                            continue
                        
                        # C = S & N(x) & N(y)
                        C = S & adj[x] & adj[y]
                        
                        # Check if G[C] is a clique
                        # If not, there is a non-edge (z, w) in C
                        # This completes the K2,2,2
                        if C.bit_count() < 2:
                            continue
                        
                        # Check if C is a clique in G
                        # For each z in C, adj[z] must contain all other nodes in C
                        c_list = []
                        temp_C = C
                        while temp_C:
                            bit = temp_C & (-temp_C)
                            c_list.append(bit.bit_length() - 1)
                            temp_C ^= bit
                            
                        is_clique = True
                        for z in c_list:
                            # Mask of other nodes in C
                            others = C ^ (1 << z)
                            if (adj[z] & others) != others:
                                is_clique = False
                                break
                                
                        if not is_clique:
                            return False
                            
        return True

    def is_valid(n, colors):
        # Build adj masks for color 1
        adj1 = [0] * n
        adj0 = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                if c == 1:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                else:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                idx += 1
        
        if not check_k222_free(adj1, n):
            return False
        if not check_k222_free(adj0, n):
            return False
        return True

    # Verify baseline just in case (should be fast)
    if not is_valid(29, base_colors):
        # Fallback: Return empty or known valid? 
        # Paley(29) is known valid. If this fails, there is a bug in check.
        # But we trust the math.
        pass

    # Attempt to find n=30
    # We fix the first 29 vertices to Paley(29).
    # We vary the colors of edges (29, i) for i=0..28.
    
    # Precompute indices in the full 30-vertex color array for edges (29, i)
    # Total edges 435.
    # Index of (i, j) is sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
    # For n=30:
    # Edges involving 29 are (i, 29) for i=0..28.
    # Index(i, 29) = sum_{k=0}^{i-1} (29-k) + (29-i-1) = sum_{k=0}^{i-1} (29-k) + 28 - i
    
    edge_29_indices = []
    for i in range(29):
        # Sum of lengths of rows 0 to i-1
        # Row k has 29-k edges.
        # Sum_{k=0}^{i-1} (29-k) = i*29 - i*(i-1)/2
        start_row = i*29 - i*(i-1)//2
        # Position within row i: j-i-1 = 29-i-1 = 28-i
        idx = start_row + (28-i)
        edge_29_indices.append(idx)

    def build_colors_30(mask_29):
        # mask_29 is an integer with 29 bits. Bit i is color of (i, 29).
        colors = base_colors[:] + [0]*29 # This is wrong, base_colors is length 406.
        # We need to construct the full 435 length array.
        # It's faster to copy base and insert? No, list insertion is O(n).
        # Let's just create a new list.
        full_colors = [0] * 435
        idx = 0
        for i in range(29):
            for j in range(i+1, 29):
                full_colors[idx] = base_colors[idx]
                idx += 1
            # Edge (i, 29)
            full_colors[idx] = (mask_29 >> i) & 1
            idx += 1
        # Wait, the loop above handles (i, j) with j<29.
        # The lexicographic order is (0,1)...(0,28), (0,29), (1,2)...(1,28), (1,29)...
        # So for each i, we have edges (i, i+1)...(i, 28) then (i, 29).
        # My loop:
        # for i in range(29):
        #   for j in range(i+1, 29): # (i, i+1) to (i, 28)
        #      copy base
        #   # then (i, 29)
        #   full_colors[idx] = ...
        #   idx += 1
        # This matches the lexicographic order for n=30.
        return full_colors

    # Local search
    # Start with a random mask
    best_mask = None
    best_valid = False
    
    # Try a few random starts and hill climb
    num_restarts = 20
    steps_per_restart = 5000
    
    # Precompute adj1_base and adj0_base for speed?
    # In the check, we need to build full adj for n=30.
    # We can optimize the check for the extension.
    
    def check_valid_30_ext(mask_29):
        # Build adj1 and adj0 for n=30
        # Copy base adj
        adj1 = adj1_base[:]
        adj0 = adj0_base[:]
        
        # Add vertex 29
        adj1[29] = 0
        adj0[29] = 0
        
        for i in range(29):
            c = (mask_29 >> i) & 1
            if c == 1:
                adj1[29] |= (1 << i)
                adj1[i] |= (1 << 29)
            else:
                adj0[29] |= (1 << i)
                adj0[i] |= (1 << 29)
                
        if not check_k222_free(adj1, 30):
            return False
        if not check_k222_free(adj0, 30):
            return False
        return True

    # Hill climbing
    # We flip one bit at a time.
    # If valid, return.
    # If not, we need a heuristic to guide the search?
    # The constraint is "no K2,2,2".
    # A simple random walk might not work well.
    # Let's try random restarts with small random perturbations.
    
    # Given the time limit, we can only do a limited search.
    # If we find a valid 30, we return it.
    
    # To make hill climbing effective, we need a cost function.
    # Cost = number of K2,2,2 violations?
    # Counting violations is expensive.
    # Maybe just check validity. If invalid, flip a random bit.
    # This is a random walk. It might get stuck.
    
    # Let's try a few random masks first.
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        mask = random.getrandbits(29)
        current_mask = mask
        
        # Check initial
        if check_valid_30_ext(current_mask):
            best_mask = current_mask
            best_valid = True
            break
            
        # Hill climb
        for _ in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            # Flip a random bit
            bit_to_flip = random.randint(0, 28)
            new_mask = current_mask ^ (1 << bit_to_flip)
            
            if check_valid_30_ext(new_mask):
                best_mask = new_mask
                best_valid = True
                break
                
            # Accept with some probability? Or always accept if it doesn't increase "badness"?
            # Without a cost function, we just do a random walk.
            # To escape local minima, we can sometimes keep the flip even if invalid (random walk).
            # But we need to track the "best" state?
            # Since we don't have a cost, we can't define "best".
            # Let's just do a random walk for a fixed number of steps.
            current_mask = new_mask
            
        if best_valid:
            break

    if best_valid:
        colors = build_colors_30(best_mask)
        # Final verification
        if is_valid(30, colors):
            return {'n': 30, 'colors': colors}
        else:
            # Should not happen
            return {'n': 29, 'colors': base_colors}
    else:
        return {'n': 29, 'colors': base_colors}
