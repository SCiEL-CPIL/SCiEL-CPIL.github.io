<<<<<<< SEARCH
import time
import random
import sys

def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin before 90s kill

    # Precompute all 4-term APs (0-indexed)
    # AP is (i, i+d, i+2d, i+3d)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))

    # For each position, list indices of APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    def count_violations(colors):
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                v += 1
        return v

    def is_ap_valid(colors, ap):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c0, c1, c2, c3}) >= 3

    def local_search(k, time_limit):
        """
        Randomized local search to find a valid coloring with k colors.
        Returns the coloring if found, else None.
        """
        # Initialize random
        colors = [random.randint(0, k - 1) for _ in range(n)]
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = current_violations
        
        # Simulated Annealing parameters
        temp = 1.0
        cooling = 0.999
        min_temp = 0.01
        
        while time.time() < time_limit:
            # Pick a random position
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Try all other colors
            best_new_color = old_color
            best_delta = 0
            
            # Calculate current violations for APs containing pos
            affected_aps = pos_to_aps[pos]
            
            # Count how many affected APs are currently invalid
            currently_invalid_count = 0
            for ap_idx in affected_aps:
                ap = aps[ap_idx]
                if not is_ap_valid(colors, ap):
                    currently_invalid_count += 1
            
            # Try each new color
            for new_color in range(k):
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                # Recount violations for affected APs
                new_invalid_count = 0
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    if not is_ap_valid(colors, ap):
                        new_invalid_count += 1
                
                delta = new_invalid_count - currently_invalid_count
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
                elif delta == best_delta and random.random() < 0.5:
                    best_new_color = new_color
            
            # Revert to old color for now, then apply best
            colors[pos] = old_color
            
            # Decide whether to accept
            if best_delta < 0:
                accept = True
            elif best_delta == 0:
                accept = True
            else:
                # Accept worse with probability exp(-delta / temp)
                if temp > 0:
                    import math
                    prob = math.exp(-best_delta / temp)
                    accept = random.random() < prob
                else:
                    accept = False
            
            if accept:
                colors[pos] = best_new_color
                current_violations += best_delta
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                
                if current_violations == 0:
                    return colors
            
            temp *= cooling
            if temp < min_temp:
                break
        
        return best_colors if best_violations == 0 else None

    def greedy_construct():
        """
        Deterministic greedy: assign smallest valid color at each position.
        This uses the structural idea from inspiration 1.
        """
        colors = [None] * n
        for x in range(n):
            found = False
            for color in range(128):
                valid = True
                # Check all APs that include position x
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    # Build the set of colors in this AP, replacing x's color with candidate
                    cset = set()
                    for pos in ap:
                        if pos == x:
                            cset.add(color)
                        else:
                            if colors[pos] is None:
                                valid = False
                                break
                            cset.add(colors[pos])
                    if not valid:
                        break
                    if len(cset) < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = color
                    found = True
                    break
            if not found:
                # Fallback: should not happen with 128 colors
                colors[x] = x
        return colors

    # Try the greedy construction first
    greedy_result = greedy_construct()
    greedy_violations = count_violations(greedy_result)
    greedy_colors_used = len(set(greedy_result))
    
    best_overall = greedy_result
    best_overall_colors = greedy_colors_used
    
    if greedy_violations == 0:
        # We have a valid solution, try to reduce colors
        # Use local search to try to reduce the number of colors
        for k in range(greedy_colors_used, 2, -1):
            if time.time() > deadline - 5.0:
                break
            
            # Try to find a valid coloring with k colors
            time_slice = 5.0
            result = local_search(k, time.time() + time_slice)
            if result is not None:
                best_overall = result
                best_overall_colors = k
                break
    
    # If greedy didn't work, or we couldn't reduce, try local search with increasing k
    if best_overall_colors > 3:
        k_values = [3, 4, 5, 6, 8, 10, 16, 32, 64, 128]
        
        for k in k_values:
            if time.time() > deadline - 5.0:
                break
            
            # Give each k a time slice, but prioritize smaller k
            time_slice = 10.0 if k <= 4 else 5.0
            if time.time() + time_slice > deadline - 2.0:
                time_slice = max(1.0, deadline - 2.0 - time.time())
            
            result = local_search(k, time.time() + time_slice)
            if result is not None:
                best_overall = result
                best_overall_colors = k
                break
    
    # Fallback: Identity mapping (always valid, uses 128 colors)
    if best_overall_colors == 128:
        return list(range(n))
    
    return best_overall
=======
import time
import random
import sys
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin before 90s kill

    # Precompute all 4-term APs (0-indexed)
    # AP is (i, i+d, i+2d, i+3d)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))

    num_aps = len(aps)

    # For each position, list indices of APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    # Precompute AP color masks for fast violation counting
    # For each AP, store the 4 positions
    ap_positions = [list(ap) for ap in aps]

    def count_violations(colors):
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                v += 1
        return v

    def is_ap_valid(colors, ap):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c0, c1, c2, c3}) >= 3

    def get_violated_aps(colors):
        """Return list of indices of violated APs."""
        violated = []
        for idx, ap in enumerate(aps):
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                violated.append(idx)
        return violated

    def walksat_search(k, time_limit, rng_seed):
        """
        WalkSAT-style: pick a violated clause (AP), flip a random position in it.
        Mix of greedy (pick flip that reduces violations most) and random.
        """
        rng = random.Random(rng_seed)
        colors = [rng.randint(0, k - 1) for _ in range(n)]
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = current_violations
        stuck_count = 0
        max_stuck = 500
        
        while time.time() < time_limit:
            if current_violations == 0:
                return colors
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                stuck_count = 0
            else:
                stuck_count += 1
            
            if stuck_count > max_stuck:
                # Restart with new random coloring
                colors = [rng.randint(0, k - 1) for _ in range(n)]
                current_violations = count_violations(colors)
                stuck_count = 0
                continue
            
            # Get violated APs
            violated = get_violated_aps(colors)
            if not violated:
                return colors
            
            # Pick a random violated AP
            ap_idx = violated[rng.randint(0, len(violated) - 1)]
            ap = aps[ap_idx]
            
            # Try flipping each position in this AP
            best_delta = float('inf')
            best_pos = -1
            best_color = -1
            deltas = []
            
            for pos in ap:
                old_color = colors[pos]
                for new_color in range(k):
                    if new_color == old_color:
                        continue
                    colors[pos] = new_color
                    # Count violations for APs containing pos
                    new_viol = 0
                    for ap2_idx in pos_to_aps[pos]:
                        ap2 = aps[ap2_idx]
                        c0, c1, c2, c3 = colors[ap2[0]], colors[ap2[1]], colors[ap2[2]], colors[ap2[3]]
                        if len({c0, c1, c2, c3}) < 3:
                            new_viol += 1
                    colors[pos] = old_color
                    # We need delta relative to current state
                    # Approximate: just track if this flip helps this specific AP
                    deltas.append((new_viol, pos, new_color))
            
            # Sort by new violation count, pick from top with some randomness
            deltas.sort(key=lambda x: x[0])
            
            # Greedy vs random mix: 70% pick best, 30% pick random from top 3
            if rng.random() < 0.7 and deltas:
                _, pos, new_color = deltas[0]
            elif deltas:
                top_k = min(3, len(deltas))
                _, pos, new_color = deltas[rng.randint(0, top_k - 1)]
            else:
                continue
            
            colors[pos] = new_color
            current_violations = count_violations(colors)
        
        return best_colors if best_violations == 0 else None

    def tabu_search(k, time_limit, rng_seed):
        """
        Tabu search with short tabu list.
        """
        rng = random.Random(rng_seed + 1000)
        colors = [rng.randint(0, k - 1) for _ in range(n)]
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = current_violations
        tabu_list = {}  # (pos, old_color) -> iteration when tabu expires
        tabu_tenure = 20
        iteration = 0
        stuck_count = 0
        max_stuck = 800
        
        while time.time() < time_limit:
            iteration += 1
            if current_violations == 0:
                return colors
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                stuck_count = 0
            else:
                stuck_count += 1
            
            if stuck_count > max_stuck:
                # Restart
                colors = [rng.randint(0, k - 1) for _ in range(n)]
                current_violations = count_violations(colors)
                tabu_list = {}
                iteration = 0
                stuck_count = 0
                continue
            
            # Pick a random position, try all colors, respect tabu
            pos = rng.randint(0, n - 1)
            old_color = colors[pos]
            
            best_delta = float('inf')
            best_new_color = -1
            
            for new_color in range(k):
                if new_color == old_color:
                    continue
                # Check tabu
                key = (pos, new_color)
                if key in tabu_list and tabu_list[key] > iteration:
                    # Aspiration: allow if it improves best
                    colors[pos] = new_color
                    new_viol = count_violations(colors)
                    colors[pos] = old_color
                    if new_viol > best_violations:
                        continue
                
                colors[pos] = new_color
                new_viol = count_violations(colors)
                delta = new_viol - current_violations
                colors[pos] = old_color
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
            
            if best_new_color == -1:
                # All tabu, pick random
                best_new_color = rng.randint(0, k - 1)
                if best_new_color == old_color:
                    best_new_color = (best_new_color + 1) % k
                best_delta = 0  # approximate
            
            colors[pos] = best_new_color
            tabu_list[(pos, best_new_color)] = iteration + tabu_tenure
            current_violations = count_violations(colors)
        
        return best_colors if best_violations == 0 else None

    def sa_search(k, time_limit, rng_seed):
        """
        Simulated annealing with different parameters.
        """
        rng = random.Random(rng_seed + 2000)
        colors = [rng.randint(0, k - 1) for _ in range(n)]
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = current_violations
        
        temp = 2.0
        cooling = 0.9995
        min_temp = 0.05
        
        while time.time() < time_limit:
            pos = rng.randint(0, n - 1)
            old_color = colors[pos]
            
            best_delta = 0
            best_new_color = old_color
            
            for new_color in range(k):
                if new_color == old_color:
                    continue
                colors[pos] = new_color
                new_viol = count_violations(colors)
                delta = new_viol - current_violations
                colors[pos] = old_color
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
            
            if best_delta < 0 or (best_delta == 0 and rng.random() < 0.3):
                colors[pos] = best_new_color
                current_violations += best_delta
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                if current_violations == 0:
                    return colors
            elif temp > min_temp:
                prob = math.exp(-best_delta / temp)
                if rng.random() < prob:
                    colors[pos] = best_new_color
                    current_violations += best_delta
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_colors = colors[:]
                    if current_violations == 0:
                        return colors
            
            temp *= cooling
            if temp < min_temp:
                # Restart with fresh random
                colors = [rng.randint(0, k - 1) for _ in range(n)]
                current_violations = count_violations(colors)
                temp = 2.0
                if current_violations == 0:
                    return colors
        
        return best_colors if best_violations == 0 else None

    def portfolio_search(k, time_limit):
        """
        Run multiple search strategies in parallel threads.
        Return the best valid coloring found, or None.
        """
        results = [None]
        result_lock = threading.Lock()
        
        def run_search(strategy_name, seed_offset):
            """Run one search strategy, store result if valid."""
            local_result = None
            if strategy_name == "walksat":
                local_result = walksat_search(k, time_limit, seed_offset)
            elif strategy_name == "tabu":
                local_result = tabu_search(k, time_limit, seed_offset)
            elif strategy_name == "sa":
                local_result = sa_search(k, time_limit, seed_offset)
            
            if local_result is not None:
                with result_lock:
                    if results[0] is None:
                        results[0] = local_result
        
        # Launch multiple threads with different strategies and seeds
        strategies = []
        base_seed = int(time.time() * 1000) % (2**31)
        
        # 4 WalkSAT instances
        for i in range(4):
            strategies.append(("walksat", base_seed + i * 100))
        # 3 Tabu instances
        for i in range(3):
            strategies.append(("tabu", base_seed + 1000 + i * 100))
        # 3 SA instances
        for i in range(3):
            strategies.append(("sa", base_seed + 2000 + i * 100))
        
        with ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            futures = []
            for name, seed in strategies:
                futures.append(executor.submit(run_search, name, seed))
            
            # Wait for any to complete with a valid result
            for future in as_completed(futures):
                if results[0] is not None:
                    break
                # Just wait for completion
                try:
                    future.result()
                except Exception:
                    pass
        
        return results[0]

    def greedy_construct():
        """
        Deterministic greedy: assign smallest valid color at each position.
        """
        colors = [None] * n
        for x in range(n):
            found = False
            for color in range(128):
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    cset = set()
                    for pos in ap:
                        if pos == x:
                            cset.add(color)
                        else:
                            if colors[pos] is None:
                                valid = False
                                break
                            cset.add(colors[pos])
                    if not valid:
                        break
                    if len(cset) < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = color
                    found = True
                    break
            if not found:
                colors[x] = x
        return colors

    # Try greedy construction first
    greedy_result = greedy_construct()
    greedy_violations = count_violations(greedy_result)
    greedy_colors_used = len(set(greedy_result))
    
    best_overall = greedy_result
    best_overall_colors = greedy_colors_used
    
    # Try to find valid colorings with small k using parallel portfolio
    k_values = [3, 4, 5, 6, 7, 8]
    
    for k in k_values:
        if time.time() > deadline - 10.0:
            break
        
        # Time budget per k value
        time_slice = min(12.0, deadline - 10.0 - time.time())
        if time_slice < 2.0:
            break
        
        result = portfolio_search(k, time.time() + time_slice)
        if result is not None:
            # Verify
            if count_violations(result) == 0:
                best_overall = result
                best_overall_colors = k
                break
    
    # If we haven't found a small-k solution, try larger k
    if best_overall_colors > 8:
        for k in [10, 12, 16, 20]:
            if time.time() > deadline - 5.0:
                break
            time_slice = min(8.0, deadline - 5.0 - time.time())
            if time_slice < 1.0:
                break
            result = portfolio_search(k, time.time() + time_slice)
            if result is not None and count_violations(result) == 0:
                best_overall = result
                best_overall_colors = k
                break
    
    # Fallback: identity mapping (always valid, uses 128 colors)
    if best_overall_colors >= 128:
        return list(range(n))
    
    return best_overall
>>>>>>> REPLACE
