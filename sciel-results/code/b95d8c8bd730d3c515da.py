import time
import random

N = 16
MAX_COMPARATORS = 4096

# Precompute valid sorted masks for zero-one principle check
# A 16-bit integer is sorted (non-decreasing) if it is of the form 0...01...1
VALID_SORTED_MASKS = set()
for k in range(N + 1):
    VALID_SORTED_MASKS.add((1 << k) - 1)

def is_valid_network(network):
    """
    Verifies if the network sorts all 2^N inputs using the zero-one principle.
    Returns True if valid, False otherwise.
    """
    for mask in range(1 << N):
        m = mask
        for i, j in network:
            # Extract bits
            bi = (m >> i) & 1
            bj = (m >> j) & 1
            if bi != bj:
                if bi > bj:
                    # Swap: i gets 0, j gets 1
                    # Currently i has 1, j has 0
                    m ^= (1 << i)
                    m ^= (1 << j)
                # else: i has 0, j has 1, which is already min/max order
            # If bi == bj, no change needed
        if m not in VALID_SORTED_MASKS:
            return False
    return True

def generate_bitonic_network(n):
    """
    Generates a standard bitonic sorting network for n wires.
    Complexity: O(n log^2 n) comparators.
    For n=16, this yields 80 comparators.
    """
    if n <= 1:
        return []
    
    # Bitonic sort recursive structure
    # Split into two halves, sort each in opposite orders, then merge
    # Standard iterative bitonic sort is easier to implement correctly
    
    def bitonic_merge(comp_list, start, length, ascending):
        if length <= 1:
            return
        half = length // 2
        # Recursively sort the two halves in opposite directions
        bitonic_merge(comp_list, start, half, ascending)
        bitonic_merge(comp_list, start + half, half, not ascending)
        
        # Compare-exchange pairs
        for i in range(half):
            if ascending:
                comp_list.append((start + i, start + i + half))
            else:
                comp_list.append((start + i + half, start + i)) # Ensure i < j for the pair format? 
                # Wait, the task requires i < j.
                # If ascending is False, we want max at i, min at j?
                # Standard compare-exchange (i, j) puts min at i, max at j.
                # In bitonic sort, when merging two sorted sequences (one asc, one desc) into one asc:
                # We compare a[i] and a[i+half]. If a[i] > a[i+half], swap.
                # This results in min at i, max at i+half.
                # So (i, i+half) is always the correct comparator operation regardless of the "direction" flag 
                # IF we define the merge step correctly.
                
                # Let's re-verify standard bitonic merge.
                # To merge two bitonic sequences of length L into one sorted ascending sequence:
                # 1. Sort first half ascending, second half descending.
                # 2. Compare a[i] with a[i+L/2].
                #    If a[i] > a[i+L/2], swap.
                #    This puts smaller element at i, larger at i+L/2.
                #    This is exactly what comparator (i, i+L/2) does.
                
                # So the comparator is always (start + i, start + i + half) with i < j.
                # The "direction" only affects the recursive sorting of the halves.
                pass
        # Correction: The loop above should just add the comparators.
        # I will rewrite the generation cleanly.

    # Clean implementation of Bitonic Sort Network
    comps = []
    
    def bitonic_sort(start, length, ascending):
        if length <= 1:
            return
        half = length // 2
        # Recursively sort halves in opposite directions
        bitonic_sort(start, half, ascending)
        bitonic_sort(start + half, half, not ascending)
        
        # Merge phase
        for i in range(half):
            # Compare (start + i) and (start + i + half)
            # Comparator (a, b) sorts a and b such that a gets min, b gets max.
            # This is exactly what we need for the merge step.
            comps.append((start + i, start + i + half))

    bitonic_sort(0, n, True)
    return comps

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 seconds safety margin for 90s limit
    
    # 1. Initialize with Bitonic Sort Network (80 comparators for N=16)
    current_net = generate_bitonic_network(N)
    best_net = current_net
    best_len = len(current_net)
    
    # Verify initial network (should be valid)
    if not is_valid_network(current_net):
        # Fallback to Odd-Even Transposition if Bitonic gen is flawed (unlikely)
        current_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        best_net = current_net
        best_len = len(current_net)
        
    # 2. Local Search / Simulated Annealing
    # We will primarily try to remove comparators.
    # We can also try swapping or adding, but removal is the key to improvement.
    
    # To speed up verification, we can't easily cache results because the network changes.
    # However, 65,536 checks per validation is fast in Python if optimized?
    # Let's estimate: 65536 * 80 ops ~ 5M ops. In Python, ~0.1s - 0.5s.
    # So we can do maybe 10-100 validations per second.
    # In 80 seconds, we can do ~1000-8000 iterations.
    
    # Optimization: Instead of checking ALL 65536 inputs every time, 
    # we can maintain a set of "witnesses" - inputs that failed previously.
    # If a new network passes all current witnesses, we might still be invalid.
    # But if it fails a witness, it's definitely invalid.
    # Strategy:
    # 1. Keep a list of failing inputs from previous invalid attempts.
    # 2. When validating a new net, first check against the list of known failing inputs.
    # 3. If it passes those, do a full check.
    # 4. If full check fails, add the new failing input to the list.
    # 5. If full check passes, the network is valid.
    
    failing_inputs = []
    
    def quick_check(net):
        """Check against known failing inputs first."""
        for mask in failing_inputs:
            m = mask
            for i, j in net:
                bi = (m >> i) & 1
                bj = (m >> j) & 1
                if bi != bj and bi > bj:
                    m ^= (1 << i)
                    m ^= (1 << j)
            if m not in VALID_SORTED_MASKS:
                return False, mask
        return True, None

    def full_check(net):
        """Full zero-one principle check."""
        for mask in range(1 << N):
            m = mask
            for i, j in net:
                bi = (m >> i) & 1
                bj = (m >> j) & 1
                if bi != bj and bi > bj:
                    m ^= (1 << i)
                    m ^= (1 << j)
            if m not in VALID_SORTED_MASKS:
                return False, mask
        return True, None

    # Initial full check is already done implicitly by assuming Bitonic is valid.
    # But let's be safe.
    valid, _ = full_check(current_net)
    if not valid:
        # If bitonic is invalid (shouldn't happen), fall back to OET
        current_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        best_net = current_net
        best_len = len(current_net)
        valid, _ = full_check(current_net)
        if not valid:
            return best_net # Should not happen

    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Mutation Strategy
        # 1. Try removing a random comparator (most likely to improve)
        # 2. Try swapping two comparators (neutral length, might help future removals)
        # 3. Try adding a comparator (rarely useful for minimization, but might escape local minima)
        
        mutation_type = random.random()
        
        new_net = list(current_net)
        
        if mutation_type < 0.7:
            # Remove
            if len(new_net) > 0:
                idx = random.randint(0, len(new_net) - 1)
                new_net.pop(idx)
            else:
                continue
        elif mutation_type < 0.9:
            # Swap
            if len(new_net) >= 2:
                idx1 = random.randint(0, len(new_net) - 1)
                idx2 = random.randint(0, len(new_net) - 1)
                new_net[idx1], new_net[idx2] = new_net[idx2], new_net[idx1]
            else:
                continue
        else:
            # Add
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            new_net.append((i, j))
            
        # Validate
        # First, quick check against known failing inputs
        ok, fail_mask = quick_check(new_net)
        
        if not ok:
            # If it fails a known bad input, it's invalid.
            # We can update current_net to this invalid one? No, keep current valid.
            # But we can add the fail_mask to failing_inputs if it's new? 
            # It's already in failing_inputs if quick_check found it.
            continue
            
        # If quick check passed, do full check
        ok, fail_mask = full_check(new_net)
        
        if ok:
            # Valid network found
            if len(new_net) < best_len:
                best_net = new_net
                best_len = len(new_net)
                current_net = new_net
                # Reset failing inputs? No, keep them, they are still valid witnesses for invalidity.
                # Actually, if we found a valid net, the previous failing inputs were for *other* nets.
                # We don't need to reset, but the list might grow.
                # It's fine.
            else:
                # Same length or worse.
                # Accept same length with some probability to explore
                if len(new_net) == len(current_net) and random.random() < 0.1:
                    current_net = new_net
                # Reject worse
        else:
            # Invalid network. Add the failing mask to our list of witnesses.
            if fail_mask not in failing_inputs:
                failing_inputs.append(fail_mask)
                # Cap the list to prevent memory bloat or slow quick checks
                if len(failing_inputs) > 1000:
                    # Randomly drop some old ones? Or just keep the most recent?
                    # Let's just truncate
                    failing_inputs = failing_inputs[-500:]

    return best_net

if __name__ == "__main__":
    # Simple self-test
    net = construct()
    print(f"Found network with {len(net)} comparators.")
    # Verify one more time
    valid, _ = full_check(net)
    print(f"Valid: {valid}")
