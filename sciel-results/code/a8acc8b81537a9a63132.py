import time
import random
import sys

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Build the standard superpermutation for n-1 and extract the permutation order
    def build_superperm_order(m):
        """Build standard superpermutation for m symbols, return list of permutations in order."""
        if m == 1:
            return [(1,)]
        if m == 2:
            return [(1,2), (2,1)]
        
        prev_perms = build_superperm_order(m-1)
        
        # Build the word for m
        word = []
        for p in prev_perms:
            block = list(p) + [m] + list(p)
            # Find max overlap between current word tail and block head
            max_overlap = min(len(word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if word[-o:] == block[:o]:
                    overlap = o
                    break
            word += block[overlap:]
        
        # Extract unique permutations of length m in order of first appearance
        perms = []
        seen = set()
        for i in range(len(word) - m + 1):
            window = tuple(word[i:i+m])
            if window not in seen:
                seen.add(window)
                perms.append(window)
        
        return perms

    perms = build_superperm_order(n-1)
    num_perms = len(perms)
    
    # Precompute overlap matrix efficiently
    # overlap[i][j] = max k in [0, n-2] such that perms[i][-k:] == perms[j][:k]
    max_k = n - 2
    overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
    
    # For each pair, check from largest k down
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            for k in range(max_k, 0, -1):
                if pi[-k:] == pj[:k]:
                    overlap_matrix[i][j] = k
                    break
            # if no overlap found, stays 0
    
    block_len = 2 * (n - 1) + 1
    
    # Total length = block_len * num_perms - sum of overlaps between consecutive
    # So minimizing length = maximizing sum of overlaps
    def calc_total_overlap(ord_list):
        total = 0
        for idx in range(len(ord_list) - 1):
            total += overlap_matrix[ord_list[idx]][ord_list[idx+1]]
        return total
    
    # Initial order
    order = list(range(num_perms))
    best_overlap = calc_total_overlap(order)
    best_order = order[:]
    
    # 2-opt local search with simulated annealing
    # A move: reverse a segment [i, j]
    # Delta in overlap = change at boundaries (i-1, i), (j, j+1), and internal
    
    def get_delta_reverse(ord_list, i, j):
        """Compute delta in total overlap when reversing segment [i, j]."""
        if i >= j:
            return 0
        # Before: ... ord[i-1] -> ord[i] ... ord[j] -> ord[j+1] ...
        # After:  ... ord[i-1] -> ord[j] ... ord[i] -> ord[j+1] ...
        # Internal: all pairs within [i,j] are reversed, but the sum of overlaps
        # for pairs (a,b) where both in [i,j] changes to overlaps of reversed pairs.
        # Actually for 2-opt on TSP-like, we need to account for all internal changes.
        # But that's O(n) per move. For n=720, that's manageable if we're smart.
        
        # Delta = -ov[ord[i-1]][ord[i]] - ov[ord[j]][ord[j+1]] 
        #        + ov[ord[i-1]][ord[j]] + ov[ord[i]][ord[j+1]]
        #        + (sum of ov[ord[k]][ord[k+1]] for k in [i, j-1]) 
        #        - (sum of ov[ord[k]][ord[k-1]] for k in [i+1, j])
        # The internal sums: sum_{k=i}^{j-1} ov[ord[k]][ord[k+1]] vs sum_{k=i+1}^{j} ov[ord[k]][ord[k-1]]
        # These are the same set of unordered pairs but with reversed direction.
        # ov[a][b] != ov[b][a] in general.
        
        old_internal = 0
        for k in range(i, j):
            old_internal += overlap_matrix[ord_list[k]][ord_list[k+1]]
        
        new_internal = 0
        for k in range(i, j):
            # After reversal, position k has ord_list[j-k+i]
            # The pair at positions k, k+1 after reversal:
            # pos k: ord_list[j - (k - i)] = ord_list[j - k + i]
            # pos k+1: ord_list[j - (k+1 - i)] = ord_list[j - k + i - 1]
            a = ord_list[j - k + i]
            b = ord_list[j - k + i - 1]
            new_internal += overlap_matrix[a][b]
        
        delta = 0
        if i > 0:
            delta -= overlap_matrix[ord_list[i-1]][ord_list[i]]
            delta += overlap_matrix[ord_list[i-1]][ord_list[j]]
        if j < len(ord_list) - 1:
            delta -= overlap_matrix[ord_list[j]][ord_list[j+1]]
            delta += overlap_matrix[ord_list[i]][ord_list[j+1]]
        delta += new_internal - old_internal
        
        return delta
    
    # Simulated annealing 2-opt
    current_overlap = best_overlap
    current_order = order[:]
    temperature = 10.0
    cooling_rate = 0.9995
    min_temp = 0.1
    
    rng = random.Random(42)
    
    while time.time() < deadline - 10.0:
        # Try a 2-opt move
        i = rng.randint(0, num_perms - 2)
        j = rng.randint(i + 1, num_perms - 1)
        # Limit segment length for efficiency in early stages
        if j - i > 50 and temperature > 5.0:
            j = min(j, i + 50)
        
        delta = get_delta_reverse(current_order, i, j)
        
        if delta >= 0 or rng.random() < pow(2.718281828, delta / max(temperature, 0.01)):
            # Accept
            current_order[i:j+1] = current_order[i:j+1][::-1]
            current_overlap += delta
            
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_order = current_order[:]
        
        temperature *= cooling_rate
        if temperature < min_temp:
            # Reset with a perturbation
            temperature = 10.0
            # Random perturbation: reverse a random segment
            i = rng.randint(0, num_perms - 2)
            j = rng.randint(i + 1, min(i + 30, num_perms - 1))
            current_order[i:j+1] = current_order[i:j+1][::-1]
            current_overlap = calc_total_overlap(current_order)
    
    # Final local improvement: greedy 2-opt pass
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(num_perms - 1):
            for j in range(i + 1, min(i + 20, num_perms)):
                delta = get_delta_reverse(best_order, i, j)
                if delta > 0:
                    best_order[i:j+1] = best_order[i:j+1][::-1]
                    best_overlap += delta
                    improved = True
    
    # Construct final string
    final_word = []
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        max_overlap = min(len(final_word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if o == 0:
                overlap = 0
                break
            if final_word[-o:] == block[:o]:
                overlap = o
                break
        final_word += block[overlap:]
    
    return "".join(map(str, final_word))
