"""Size-efficient sorting network on 17 wires via bitmask-verified greedy removal."""
import time
import random

N = 17
TOTAL_MASK = (1 << N) - 1  # 17 bits all set

# Precompute bit positions for fast access
# wire i corresponds to bit i in the mask

def _apply_comparator(mask, i, j):
    """Apply compare-exchange to a single bitmask state.
    If bit i is 1 and bit j is 0, swap them (min to i, max to j).
    """
    bit_i = (mask >> i) & 1
    bit_j = (mask >> j) & 1
    if bit_i == 1 and bit_j == 0:
        # Need to swap: clear bit i, set bit j
        mask ^= (1 << i) | (1 << j)
    return mask

def _is_sorted_mask(mask):
    """Check if a bitmask represents a sorted sequence (non-decreasing from wire 0 to N-1).
    Sorted means: all 1s are on the right (higher wire indices), all 0s on the left.
    So mask should be of the form: 000...0111...1
    """
    # A sorted mask has the property that once you see a 1, all subsequent bits are 1.
    # Equivalent: mask == ((1 << N) - 1) ^ ((1 << k) - 1) for some k in [0, N]
    # i.e., mask is a suffix of 1s.
    # Check: mask & (mask + 1) == 0 means mask is of form 00..011..1 (a contiguous block of 1s at the bottom? No.)
    # Actually for 00011110, that's not a suffix of 1s from the LSB.
    # Let me reconsider. Wire 0 is the smallest output. Sorted ascending means wire 0 <= wire 1 <= ... <= wire N-1.
    # For boolean inputs, sorted means all 0s come first, then all 1s.
    # So the mask should be: bits 0..k-1 are 0, bits k..N-1 are 1, for some k in [0, N].
    # That is: mask = TOTAL_MASK ^ ((1 << k) - 1) for k in [0, N].
    # Equivalently: mask has the form where if we look at the binary representation,
    # it's a sequence of 0s followed by 1s.
    
    # Quick check: mask should satisfy that for all i < j, if bit i is 1 then bit j is 1.
    # This is equivalent to: mask & (mask << 1) ... no.
    # Simplest: count the number of 1s, then check if mask equals the lower-popcount bits set.
    # popcount = number of 1s. If sorted, mask should be (1 << popcount) - 1? No, that's 1s at the bottom.
    # Wait: wire 0 is index 0 (LSB). Sorted ascending: 0,0,...,0,1,1,...,1
    # So the 1s are in the higher-index wires. In binary (LSB = wire 0), this means
    # the lower k bits are 0 and the upper N-k bits are 1.
    # So mask = TOTAL_MASK & ~((1 << k) - 1) = TOTAL_MASK ^ ((1 << k) - 1)
    
    # Check: if we remove the lowest set bit repeatedly, we should never encounter a 0 before a 1.
    # Actually, the simplest check: mask must be of the form where all bits below the highest 0 are 0.
    # Let's just check: for the mask to be sorted (0s then 1s from wire 0 to N-1),
    # it must be that mask == ((1 << N) - 1) - ((1 << k) - 1) for some k.
    # That equals (1 << N) - 1 - (1 << k) + 1 = (1 << N) - (1 << k).
    # Hmm, let me just do it directly.
    
    # A mask is "sorted" (ascending) iff: there exists k such that bits 0..k-1 are 0 and bits k..N-1 are 1.
    # This means: mask & ((1 << k) - 1) == 0 and mask | ((1 << k) - 1) == TOTAL_MASK.
    # Equivalently: mask must be a "suffix" of 1s in the most significant bits.
    
    # Fast check: mask should be of the form where if we write it in binary (N bits, MSB left),
    # it looks like 000...0111...1.
    # In terms of integer: mask = TOTAL_MASK - ((1 << k) - 1) = (1<<N)-1 - (1<<k)+1 = (1<<N) - (1<<k)
    # So mask + (1<<k) = (1<<N) for some k.
    # Alternatively: the number of trailing zeros in the complement... 
    
    # Simplest O(N) check but N=17 so it's fine:
    # Check that once we see a 1, all higher bits are 1.
    seen_one = False
    for i in range(N):
        if (mask >> i) & 1:
            seen_one = True
        else:
            if seen_one:
                return False
    return True

def _verify_network(comparators):
    """Verify that the network sorts all 2^17 boolean inputs using bitmask simulation.
    We test all 2^17 = 131072 inputs. For each input bitmask, apply all comparators,
    and check if the result is sorted.
    """
    for inp in range(1 << N):
        mask = inp
        for (i, j) in comparators:
            mask = _apply_comparator(mask, i, j)
        if not _is_sorted_mask(mask):
            return False
    return True

def _verify_network_fast(comparators, sample_size=None):
    """Faster verification: test all 2^N inputs but with optimized inner loop.
    For N=17, 131072 inputs is manageable.
    """
    for inp in range(1 << N):
        mask = inp
        for (i, j) in comparators:
            # Inline the comparator for speed
            if ((mask >> i) & 1) and not ((mask >> j) & 1):
                mask ^= (1 << i) | (1 << j)
        # Check sorted: all 0s before all 1s
        # mask should be of form: lower k bits 0, upper bits 1
        # Check: mask & (mask + 1) == 0 is for power of 2 or 0. Not what we want.
        # We want: mask is a suffix of 1s. 
        # Let's check: if we AND mask with (mask - 1), and the result... 
        # Actually: a number of form 00..011..1 has the property that mask & (mask+1) ... 
        # No. Let's think differently.
        # mask = 0b...01111 (1s in high bits). 
        # The complement ~mask & TOTAL_MASK = 0b...10000 (1s in low bits, 0 in high).
        # A number with 1s only in low bits is of form (1<<k)-1, which satisfies x & (x+1) == 0.
        # So: (~mask & TOTAL_MASK) & ((~mask & TOTAL_MASK) + 1) == 0
        comp = (~mask) & TOTAL_MASK
        if comp & (comp + 1) != 0:
            return False
    return True

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network for N wires."""
    comparators = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def _batcher_odd_even_merge_sort(n):
    """Generate a Batcher odd-even merge sort network for n wires.
    This is a recursive construction.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 3:
        # 3-wire sorting network: 3 comparators
        return [(0, 1), (1, 2), (0, 1)]
    if n == 4:
        # 4-wire: 5 comparators
        return [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)]
    
    # General recursive Batcher odd-even merge
    # Split into two halves, sort each, then merge
    half = n // 2
    # Sort first half (wires 0..half-1) and second half (wires half..n-1)
    left = _batcher_odd_even_merge_sort(half)
    right = _batcher_odd_even_merge_sort(n - half)
    right_shifted = [(i + half, j + half) for (i, j) in right]
    
    comparators = left + right_shifted
    
    # Now merge: odd-even merge of two sorted halves
    # For merging two sorted sequences of lengths a and b:
    # We need to perform a merge network.
    # Batcher's odd-even merge:
    # For n wires split into two halves of size half and n-half:
    # The merge step involves comparing pairs (i, i+half) for appropriate i,
    # then recursively merging.
    
    # Actually, let me use a simpler known construction for small n.
    # For n=17, let me just use a known good network.
    # Batcher's bitonic sort for 16 wires uses 60 comparators.
    # For 17, we can do bitonic sort on 16 and then merge the 17th.
    
    # Let's use a different approach: generate a known efficient network.
    # For now, return the odd-even transposition as fallback.
    return None

def _bitonic_sort_network(n):
    """Generate a bitonic sort network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Bitonic sort:
    # 1. Sort first half ascending, second half descending
    # 2. Merge the two halves
    
    # Sort first half ascending
    left = _bitonic_sort_network(half)
    # Sort second half descending: sort ascending then reverse
    right = _bitonic_sort_network(half)
    # To make it descending, we can reverse the comparator order or just note
    # that a descending sort is the reverse of ascending in terms of wire mapping.
    # Actually, for bitonic sort, we sort the second half in descending order.
    # A descending sort on wires half..n-1 can be achieved by:
    # - Sorting wires half..n-1 in ascending order (using the same network shifted)
    # - Then reversing the order of the wires? No, that's not how comparators work.
    
    # Alternative: use the fact that to sort in descending order, we can:
    # Apply the ascending sort network but with the comparison direction reversed.
    # But our comparators always put min on lower index. So to get descending,
    # we need to swap the roles.
    
    # Let me use a different formulation. For bitonic sort on n = 2^k:
    # The network is defined recursively. Let me implement it properly.
    
    # For n=4: 
    #   (0,1), (2,3)  -- sort pairs
    #   (0,2), (1,3)  -- merge to bitonic
    #   (0,3), (1,2)  -- final merge
    # That's 5 comparators.
    
    # For n=8:
    #   Sort first 4 ascending, last 4 descending
    #   Then merge
    
    # Let me implement it iteratively using the standard bitonic sort algorithm.
    
    comparators = []
    
    def _add_comparators(offset, size, direction):
        """Add comparators for sorting a segment of 'size' wires starting at 'offset'.
        direction: +1 for ascending, -1 for descending.
        """
        if size <= 1:
            return
        if size == 2:
            if direction == 1:
                comparators.append((offset, offset + 1))
            else:
                comparators.append((offset, offset + 1))
                # For descending, we still use the same comparator but the effect
                # is that min goes to lower wire. To get descending, we need max on lower wire.
                # Since our comparator puts min on lower wire, for descending we need to
                # swap the wire indices in the final output. 
                # Actually, this approach is getting complicated. Let me use a different method.
            return
        
        half_size = size // 2
        # Sort first half in ascending, second half in descending (for bitonic)
        _add_comparators(offset, half_size, 1)
        _add_comparators(offset + half_size, half_size, -1)
        
        # Now merge: compare (i, i + half_size) for i in range(half_size)
        for i in range(half_size):
            if direction == 1:
                comparators.append((offset + i, offset + i + half_size))
            else:
                comparators.append((offset + i, offset + i + half_size))
        
        # Recursively merge
        # The result of the cross comparisons creates two bitonic sequences
        # that need to be merged.
        # For a proper bitonic merge, after the cross comparisons,
        # we recursively sort the top and bottom halves.
        _add_comparators(offset, half_size, direction)
        _add_comparators(offset + half_size, half_size, direction)
    
    _add_comparators(0, n, 1)
    return comparators

def _generate_seed_network():
    """Generate a good starting network. Use odd-even transposition as baseline."""
    return _odd_even_transposition()

def _try_remove_comparator(comparators, index):
    """Try removing comparator at 'index' and check if the network still sorts.
    Returns True if removal is valid (network still sorts).
    """
    if index < 0 or index >= len(comparators):
        return False
    new_comparators = comparators[:index] + comparators[index+1:]
    return _verify_network_fast(new_comparators)

def _greedy_remove(comparators, deadline):
    """Greedy removal: repeatedly try to remove comparators, starting from the end.
    Use bitmask verification for speed.
    """
    current = list(comparators)
    removed = 0
    start_time = time.time()
    
    # Try removing from the end first (later comparators are often redundant)
    # Then try from the beginning
    # Use a strategy: try all positions, keep the first valid removal
    
    max_iterations = 100000
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > deadline - 5.0:
            break
        
        found_removal = False
        n_comps = len(current)
        
        # Try removing from various positions
        # Strategy: try positions in a pseudo-random or structured order
        # Start from the end, then middle, then beginning
        positions = list(range(n_comps - 1, -1, -1))
        
        for idx in positions:
            if time.time() > deadline - 5.0:
                break
            
            # Quick check: if this comparator is a no-op on all-zeros and all-ones,
            # it's definitely redundant (but not sufficient condition).
            # Actually, a comparator (i,j) is a no-op on all-zeros (0...0) and all-ones (1...1)
            # always, since min(0,0)=0, max(0,0)=0 and min(1,1)=1, max(1,1)=1.
            # So that check doesn't help. Let's just verify.
            
            new_comparators = current[:idx] + current[idx+1:]
            if _verify_network_fast(new_comparators):
                current = new_comparators
                found_removal = True
                removed += 1
                break
        
        if not found_removal:
            # Try a different order: from the beginning
            n_comps = len(current)
            for idx in range(n_comps):
                if time.time() > deadline - 5.0:
                    break
                
                new_comparators = current[:idx] + current[idx+1:]
                if _verify_network_fast(new_comparators):
                    current = new_comparators
                    found_removal = True
                    removed += 1
                    break
        
        if not found_removal:
            # Try swapping adjacent comparators to enable further removals
            # This is a local search move
            n_comps = len(current)
            swap_done = False
            for idx in range(n_comps - 1):
                if time.time() > deadline - 5.0:
                    break
                # Swap comparators[idx] and comparators[idx+1]
                new_comparators = current[:]
                new_comparators[idx], new_comparators[idx+1] = new_comparators[idx+1], new_comparators[idx]
                if _verify_network_fast(new_comparators):
                    # Check if this swap enables a removal
                    current = new_comparators
                    swap_done = True
                    # After swap, try to remove again
                    break
            
            if not swap_done:
                # Try reversing the order of a subset
                # Or just give up on further improvements
                break
        
        iteration += 1
    
    return current

def _local_search(comparators, deadline):
    """Local search: try various moves to reduce comparator count.
    Moves: remove one comparator, swap two adjacent comparators.
    """
    current = list(comparators)
    start_time = time.time()
    no_improvement_count = 0
    
    max_rounds = 50000
    
    for _ in range(max_rounds):
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        n_comps = len(current)
        
        # Try removals
        # Use a random order to avoid getting stuck
        indices = list(range(n_comps))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 5.0:
                break
            
            new_comparators = current[:idx] + current[idx+1:]
            if _verify_network_fast(new_comparators):
                current = new_comparators
                improved = True
                no_improvement_count = 0
                break
        
        if not improved:
            # Try swaps
            indices = list(range(n_comps - 1))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - 5.0:
                    break
                
                new_comparators = current[:]
                new_comparators[idx], new_comparators[idx+1] = new_comparators[idx+1], new_comparators[idx]
                
                # Check if swap is valid (network still sorts)
                if _verify_network_fast(new_comparators):
                    # Now try to remove a comparator after the swap
                    n_new = len(new_comparators)
                    remove_indices = list(range(n_new))
                    random.shuffle(remove_indices)
                    
                    for ridx in remove_indices:
                        if time.time() > deadline - 5.0:
                            break
                        candidate = new_comparators[:ridx] + new_comparators[ridx+1:]
                        if _verify_network_fast(candidate):
                            current = candidate
                            improved = True
                            no_improvement_count = 0
                            break
                    else:
                        # Swap didn't enable removal, revert
                        continue
                
                if improved:
                    break
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > 20:
                break
    
    return current

def construct():
    """Construct a sorting network for 17 wires with minimal comparators."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Start with odd-even transposition network (136 comparators)
    network = _odd_even_transposition()
    
    # Verify the seed network
    if not _verify_network_fast(network):
        # Should not happen, but just in case
        return network
    
    # Phase 1: Greedy removal from the end
    network = _greedy_remove(network, deadline)
    
    # Phase 2: Local search with randomization
    if time.time() < deadline - 10.0:
        network = _local_search(network, deadline)
    
    # Final verification
    if not _verify_network_fast(network):
        # Fallback to seed
        return _odd_even_transposition()
    
    return network
