import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute all 4-term arithmetic progressions
    # Store as tuples of indices (0-based)
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    total_aps = len(aps)
    
    # For efficient delta calculation for segment reversal, 
    # we can just recalculate the count for the affected APs.
    # An AP is affected if any of its indices is in [start, end].
    # To make this fast, we can precompute for each index the list of APs it belongs to.
    # But for a segment reversal, the set of affected APs is the union of APs touching any index in the segment.
    # This can be large. 
    # Alternative: Just recalculate the total count if the segment is small, or use a smarter delta.
    # Given N=256 and ~10k APs, recalculating the whole count is O(10k) which is ~0.001s in Python?
    # 10,000 checks * 4 comparisons = 40,000 ops. Very fast.
    # If we do 10,000 SA steps, that's 400 million ops. Might be too slow in pure Python.
    # Let's estimate: 10^8 ops/sec in Python is optimistic, usually 10^7.
    # So 10^4 steps * 10^4 ops/step = 10^8 ops. ~10 seconds.
    # We have 115 seconds. So we can do ~100,000 steps if we optimize.
    
    def count_mono(colors):
        count = 0
        for ap in aps:
            c = colors[ap[0]]
            if c == colors[ap[1]] and c == colors[ap[2]] and c == colors[ap[3]]:
                count += 1
        return count

    def get_delta_reversal(colors, start, end):
        """
        Calculate change in mono count if we reverse colors[start:end+1].
        We only need to check APs that have at least one index in [start, end].
        """
        if start > end:
            return 0
        
        # Identify affected APs
        # An AP is affected if any of its 4 indices is in the reversed segment.
        # We can precompute a list of APs for each index, then take the union.
        # But doing this every step is expensive.
        # Alternative: Just iterate all APs and check if they are affected.
        # Total APs is ~10,000. Iterating 10,000 APs per step is okay.
        
        # Create a set of indices in the segment for fast lookup? No, just check bounds.
        old_count = 0
        new_count = 0
        
        # To avoid copying the whole array, we can simulate the reversal for the specific APs.
        # Or, simpler: 
        # 1. Store original colors of the segment.
        # 2. Apply reversal to a copy? No, modify in place, calc, revert?
        # 3. Better: For each AP, determine its state before and after.
        
        # Let's use a helper to get color at index 'i' given the reversal (start, end)
        # If i is in [start, end], its new position is start + end - i.
        # So the color at 'i' in the "new" state is colors[start + end - i].
        # The color at 'i' in the "old" state is colors[i].
        
        for ap in aps:
            # Check if any index in ap is in [start, end]
            affected = False
            for idx in ap:
                if start <= idx <= end:
                    affected = True
                    break
            
            if not affected:
                continue
            
            # Calculate old mono status
            c0 = colors[ap[0]]
            old_mono = (c0 == colors[ap[1]] and c0 == colors[ap[2]] and c0 == colors[ap[3]])
            
            # Calculate new mono status
            # New color at ap[k] is:
            # if ap[k] in [start, end]: colors[start + end - ap[k]]
            # else: colors[ap[k]]
            
            n0 = colors[start + end - ap[0]] if start <= ap[0] <= end else colors[ap[0]]
            n1 = colors[start + end - ap[1]] if start <= ap[1] <= end else colors[ap[1]]
            n2 = colors[start + end - ap[2]] if start <= ap[2] <= end else colors[ap[2]]
            n3 = colors[start + end - ap[3]] if start <= ap[3] <= end else colors[ap[3]]
            
            new_mono = (n0 == n1 and n0 == n2 and n0 == n3)
            
            if old_mono:
                old_count += 1
            if new_mono:
                new_count += 1
        
        return new_count - old_count

    best_count = float('inf')
    best_colors = [0] * N
    
    # Simulated Annealing Parameters
    # We will run multiple SA restarts.
    
    def run_sa(initial_colors, time_limit):
        colors = initial_colors[:]
        current_count = count_mono(colors)
        
        # Temperature schedule
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.995
        
        end_time = time.time() + time_limit
        steps = 0
        
        while time.time() < end_time and T > T_min:
            # Generate move: Random segment reversal
            # Length of segment: random between 1 and N/4? 
            # Shorter segments are more local, longer are more global.
            # Let's mix: 50% single flip (len 1), 50% segment reversal (len > 1)
            if random.random() < 0.5:
                # Single flip
                idx = random.randint(0, N - 1)
                start = idx
                end = idx
            else:
                # Segment reversal
                # Pick length
                length = random.randint(2, N // 2)
                start = random.randint(0, N - length)
                end = start + length - 1
            
            delta = get_delta_reversal(colors, start, end)
            
            if delta <= 0:
                # Accept improvement or equal
                # Apply move
                colors[start:end+1] = reversed(colors[start:end+1])
                current_count += delta
            else:
                # Accept with probability exp(-delta / T)
                if random.random() < math.exp(-delta / T):
                    colors[start:end+1] = reversed(colors[start:end+1])
                    current_count += delta
            
            T *= cooling_rate
            steps += 1
            
            # Check time frequently
            if steps % 100 == 0 and time.time() > end_time:
                break
        
        return colors, current_count

    # Main Loop: Multiple Restarts
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Initialize
        # Use a mix of random and "structured" starts?
        # Random is fine.
        colors = [random.randint(0, 1) for _ in range(N)]
        
        # Determine time limit for this restart
        # Give more time to later restarts? Or equal?
        # Equal is simpler.
        # If we have 115s, and we want to do ~10-20 restarts, each gets ~5-10s.
        # Let's do dynamic: if it's the first restart, give it 5s. 
        # Actually, let's just run until deadline for the last one, and shorter for others.
        
        # Strategy: 
        # 1. Quick SA (2s)
        # 2. If improved, keep best.
        # 3. Repeat.
        
        # To ensure we finish, we calculate remaining time.
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        
        # Allocate time: 
        # If we are in the first few restarts, use 2s.
        # If we are near the end, use all remaining time.
        if restart_count < 5:
            t_alloc = 2.0
        else:
            t_alloc = remaining - 0.5 # Leave 0.5s buffer
        
        if t_alloc < 0.1:
            break
            
        colors, current_count = run_sa(colors, t_alloc)
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors[:]
            
            # If we hit 0, stop
            if best_count == 0:
                break

    return best_colors
