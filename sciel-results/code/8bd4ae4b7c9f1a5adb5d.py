import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute 3-APs and 4-APs
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
            
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map elements to APs containing them
    elem_to_3aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(i)
            
    elem_to_4aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)
    
    # Precompute 3-AP neighbor structure for fast delta calculation
    # For each element x, store (other1, other2, position) for each 3-AP containing x
    elem_3ap_neighbors = [[] for _ in range(N + 1)]
    for i, (a, b, c) in enumerate(three_aps):
        elem_3ap_neighbors[a].append((b, c, 0))
        elem_3ap_neighbors[b].append((a, c, 1))
        elem_3ap_neighbors[c].append((a, b, 2))
    
    # Precompute 4-AP neighbor structure
    elem_4ap_neighbors = [[] for _ in range(N + 1)]
    for i, (a, b, c, d) in enumerate(four_aps):
        elem_4ap_neighbors[a].append((b, c, d, 0))
        elem_4ap_neighbors[b].append((a, c, d, 1))
        elem_4ap_neighbors[c].append((a, b, d, 2))
        elem_4ap_neighbors[d].append((a, b, c, 3))

    def count_3aps_bitset(bits):
        """Count 3-APs using bitset representation (bits is an integer with bit i set if element i+1 is in set)."""
        count = 0
        for a, b, c in three_aps:
            if (bits >> a) & 1 and (bits >> b) & 1 and (bits >> c) & 1:
                count += 1
        return count

    def is_valid_add_bitset(bits, x):
        """Check if adding x to set (represented as bitset) creates a 4-AP."""
        for (o1, o2, o3, pos) in elem_4ap_neighbors[x]:
            if ((bits >> o1) & 1) and ((bits >> o2) & 1) and ((bits >> o3) & 1):
                return False
        return True

    def delta_add_bitset(bits, x):
        """Change in 3-AP count if we add x to set (bitset representation)."""
        delta = 0
        for (o1, o2, pos) in elem_3ap_neighbors[x]:
            if ((bits >> o1) & 1) and ((bits >> o2) & 1):
                delta += 1
        return delta

    def delta_remove_bitset(bits, x):
        """Change in 3-AP count if we remove x from set (bitset representation)."""
        delta = 0
        for (o1, o2, pos) in elem_3ap_neighbors[x]:
            if ((bits >> o1) & 1) and ((bits >> o2) & 1):
                delta += 1
        return -delta

    def is_valid_swap_bitset(bits, x_out, x_in):
        """Check if swapping x_out for x_in is valid (no 4-APs)."""
        for (o1, o2, o3, pos) in elem_4ap_neighbors[x_in]:
            # Check if all three others are in the set after removing x_out
            all_in = True
            x_out_in_others = False
            for y in (o1, o2, o3):
                if y == x_out:
                    x_out_in_others = True
                    continue
                if not ((bits >> y) & 1):
                    all_in = False
                    break
            if all_in and not x_out_in_others:
                return False
        return True

    def greedy_construct_bitset(random_seed):
        """Greedy construction using bitset representation."""
        random.seed(random_seed)
        bits = 0
        available = list(range(1, N + 1))
        
        while available:
            candidates = []
            for x in available:
                if is_valid_add_bitset(bits, x):
                    d = delta_add_bitset(bits, x)
                    candidates.append((d, x))
            
            if not candidates:
                break
                
            candidates.sort(key=lambda item: item[0], reverse=True)
            
            top_k = min(5, len(candidates))
            if random.random() < 0.8:
                best_x = candidates[0][1]
            else:
                best_x = random.choice([c[1] for c in candidates[:top_k]])
            
            bits |= (1 << best_x)
            available.remove(best_x)
            
        return bits

    def local_search_bitset(bits, max_iter=1000):
        """Local search using bitset representation with swap moves."""
        bits = bits
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline:
                break
                
            # Try all swaps
            for x_out in range(1, N + 1):
                if not ((bits >> x_out) & 1):
                    continue
                loss = -delta_remove_bitset(bits, x_out)  # loss is positive
                
                for x_in in range(1, N + 1):
                    if x_in == x_out or ((bits >> x_in) & 1):
                        continue
                    if not is_valid_swap_bitset(bits, x_out, x_in):
                        continue
                    
                    # Calculate gain
                    bits_without = bits & ~(1 << x_out)
                    gain = delta_add_bitset(bits_without, x_in)
                    
                    if gain > loss:
                        bits = bits_without | (1 << x_in)
                        improved = True
                        break
                if improved:
                    break
        return bits

    def parallel_tempering(num_chains=5, iterations_per_chain=200):
        """Parallel tempering with multiple chains at different temperatures."""
        # Initialize chains with different greedy constructions
        chains = []
        for i in range(num_chains):
            bits = greedy_construct_bitset(i * 1000 + 42)
            chains.append({'bits': bits, 'score': count_3aps_bitset(bits), 'temp': 1.0 + i * 0.5})
        
        # Track best
        best_bits = max(chains, key=lambda c: c['score'])['bits']
        best_score = max(c['score'] for c in chains)
        
        for it in range(iterations_per_chain):
            if time.time() > deadline:
                break
            
            for ci, chain in enumerate(chains):
                bits = chain['bits']
                temp = chain['temp']
                
                # Try a random swap
                if random.random() < 0.5:
                    # Try adding a new element
                    candidates = []
                    for x in range(1, N + 1):
                        if not ((bits >> x) & 1) and is_valid_add_bitset(bits, x):
                            d = delta_add_bitset(bits, x)
                            candidates.append((d, x))
                    if candidates:
                        candidates.sort(key=lambda item: item[0], reverse=True)
                        # Metropolis acceptance
                        if random.random() < 0.7:
                            new_x = candidates[0][1]
                        else:
                            new_x = random.choice([c[1] for c in candidates[:min(3, len(candidates))]])
                        new_bits = bits | (1 << new_x)
                        new_score = chain['score'] + delta_add_bitset(bits, new_x)
                        
                        # Accept with probability based on temperature
                        if new_score >= chain['score'] or random.random() < math.exp((new_score - chain['score']) / temp):
                            chain['bits'] = new_bits
                            chain['score'] = new_score
                else:
                    # Try a swap
                    in_set = [x for x in range(1, N + 1) if (bits >> x) & 1]
                    if in_set:
                        x_out = random.choice(in_set)
                        loss = -delta_remove_bitset(bits, x_out)
                        
                        candidates = []
                        for x_in in range(1, N + 1):
                            if x_in != x_out and not ((bits >> x_in) & 1):
                                if is_valid_swap_bitset(bits, x_out, x_in):
                                    bits_without = bits & ~(1 << x_out)
                                    gain = delta_add_bitset(bits_without, x_in)
                                    delta = gain - loss
                                    candidates.append((delta, x_in))
                        
                        if candidates:
                            candidates.sort(key=lambda item: item[0], reverse=True)
                            if random.random() < 0.7:
                                new_x = candidates[0][1]
                            else:
                                new_x = random.choice([c[1] for c in candidates[:min(3, len(candidates))]])
                            bits_without = bits & ~(1 << x_out)
                            new_bits = bits_without | (1 << new_x)
                            new_score = chain['score'] + delta_add_bitset(bits_without, new_x) - loss
                            
                            if new_score >= chain['score'] or random.random() < math.exp((new_score - chain['score']) / temp):
                                chain['bits'] = new_bits
                                chain['score'] = new_score
                
                # Update best
                if chain['score'] > best_score:
                    best_score = chain['score']
                    best_bits = chain['bits']
                
                # Swap with neighbor chain (Metropolis coupling)
                if ci > 0:
                    neighbor = chains[ci - 1]
                    # Try swap
                    if random.random() < math.exp((neighbor['score'] - chain['score']) / (neighbor['temp'] + chain['temp'])):
                        chains[ci]['bits'], chains[ci - 1]['bits'] = chains[ci - 1]['bits'], chains[ci]['bits']
                        chains[ci]['score'], chains[ci - 1]['score'] = chains[ci - 1]['score'], chains[ci]['score']
        
        return best_bits, best_score

    # Run parallel tempering
    best_bits, best_score = parallel_tempering(num_chains=6, iterations_per_chain=300)
    
    # Final local search polish
    if time.time() < deadline - 5:
        polished = local_search_bitset(best_bits, max_iter=500)
        polished_score = count_3aps_bitset(polished)
        if polished_score > best_score:
            best_bits = polished
            best_score = polished_score
    
    # Convert bitset to sorted list
    result = [x for x in range(1, N + 1) if (best_bits >> x) & 1]
    return sorted(result)
