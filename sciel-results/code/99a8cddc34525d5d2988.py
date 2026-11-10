"""B2[2] set construction via simulated annealing on a bit-vector representation."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    N = 1000
    MAX_SUM = 2 * N  # 2000
    
    # Initialize with a greedy solution
    def _greedy_init():
        """Build an initial feasible set using greedy ascending order."""
        chosen = []
        counts = [0] * (MAX_SUM + 1)
        for x in range(1, N + 1):
            # Check if x can be added
            ok = True
            if counts[2 * x] >= 2:
                ok = False
            else:
                for y in chosen:
                    if counts[x + y] >= 2:
                        ok = False
                        break
            if ok:
                chosen.append(x)
                counts[2 * x] += 1
                for y in chosen[:-1]:
                    counts[x + y] += 1
        return chosen
    
    def _build_counts_array(chosen):
        """Build sum counts as a numpy array."""
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        n = len(chosen)
        for i in range(n):
            for j in range(i, n):
                s = chosen[i] + chosen[j]
                counts[s] += 1
        return counts
    
    def _compute_violations(counts):
        """Compute total violation penalty: sum over all sums of max(0, count-2)."""
        excess = np.maximum(counts - 2, 0)
        return int(np.sum(excess))
    
    # Simulated Annealing
    def _simulated_annealing(initial_set, time_limit):
        """Run SA to improve the set."""
        chosen = list(initial_set)
        chosen_set = set(chosen)
        counts = _build_counts_array(chosen)
        current_violations = _compute_violations(counts)
        current_size = len(chosen)
        
        best_size = current_size
        best_set = list(chosen)
        best_violations = current_violations
        
        # SA parameters
        T = 10.0  # initial temperature
        cooling = 0.999  # cooling factor per step
        min_T = 0.01
        
        # Pre-compute candidate list (all numbers 1..N)
        all_numbers = list(range(1, N + 1))
        
        step = 0
        while time.time() < time_limit and T > min_T:
            step += 1
            
            # Choose a random move: flip a random bit
            x = random.choice(all_numbers)
            
            if x in chosen_set:
                # Try removing x
                # Compute delta violations
                old_viol = current_violations
                # Remove x from counts
                delta = np.zeros(MAX_SUM + 1, dtype=np.int32)
                for y in chosen:
                    s = x + y
                    delta[s] -= 1
                # The 2x sum
                delta[2 * x] -= 1
                
                new_counts = counts + delta
                # Only check affected sums
                affected_sums = [2 * x] + [x + y for y in chosen if y != x]
                # Actually, all sums involving x
                new_viol = 0
                for s in set(affected_sums):
                    old_c = counts[s]
                    new_c = new_counts[s]
                    old_excess = max(0, old_c - 2)
                    new_excess = max(0, new_c - 2)
                    new_viol += new_excess - old_excess
                
                delta_viol = new_viol
                delta_size = -1
                
                # Acceptance criterion
                if delta_viol <= 0:
                    # Better or equal in violations, and we prefer larger size
                    if delta_viol < 0 or delta_size > 0:
                        # Definitely accept
                        counts = new_counts
                        chosen_set.discard(x)
                        chosen.remove(x)
                        current_violations += delta_viol
                        current_size += delta_size
                    else:
                        # Same violations, smaller size: accept with some probability
                        if random.random() < math.exp(-delta_size / T):
                            counts = new_counts
                            chosen_set.discard(x)
                            chosen.remove(x)
                            current_violations += delta_viol
                            current_size += delta_size
                else:
                    # Worse violations
                    delta_energy = delta_viol * 10 - delta_size  # weighted
                    if random.random() < math.exp(-delta_energy / T):
                        counts = new_counts
                        chosen_set.discard(x)
                        chosen.remove(x)
                        current_violations += delta_viol
                        current_size += delta_size
            else:
                # Try adding x
                # Check feasibility first for quick rejection
                # Compute delta violations
                delta = np.zeros(MAX_SUM + 1, dtype=np.int32)
                for y in chosen:
                    s = x + y
                    delta[s] += 1
                delta[2 * x] += 1
                
                affected_sums = [2 * x] + [x + y for y in chosen]
                new_viol = 0
                for s in set(affected_sums):
                    old_c = counts[s]
                    new_c = counts[s] + delta[s]
                    old_excess = max(0, old_c - 2)
                    new_excess = max(0, new_c - 2)
                    new_viol += new_excess - old_excess
                
                delta_viol = new_viol
                delta_size = 1
                
                if delta_viol <= 0:
                    # Accept (violations don't increase, size increases)
                    counts = counts + delta
                    chosen_set.add(x)
                    chosen.append(x)
                    current_violations += delta_viol
                    current_size += delta_size
                else:
                    # Worse violations
                    delta_energy = delta_viol * 10 - delta_size
                    if random.random() < math.exp(-delta_energy / T):
                        counts = counts + delta
                        chosen_set.add(x)
                        chosen.append(x)
                        current_violations += delta_viol
                        current_size += delta_size
            
            # Track best feasible solution
            if current_violations == 0 and current_size > best_size:
                best_size = current_size
                best_set = list(chosen)
                best_violations = 0
            
            # Cool down
            T *= cooling
            
            # Periodic restart: if stuck, try to repair
            if step % 1000 == 0 and current_violations > 0:
                # Try to remove elements that contribute most to violations
                # Find sums with highest excess
                excess_sums = np.where(counts > 2)[0]
                if len(excess_sums) > 0:
                    # Try removing random elements that are in over-represented sums
                    for _ in range(min(5, len(chosen))):
                        if not chosen:
                            break
                        idx = random.randint(0, len(chosen) - 1)
                        x = chosen[idx]
                        # Check if removing helps
                        delta = np.zeros(MAX_SUM + 1, dtype=np.int32)
                        for y in chosen:
                            if y != x:
                                s = x + y
                                delta[s] -= 1
                        delta[2 * x] -= 1
                        
                        new_viol = 0
                        affected = [2 * x] + [x + y for y in chosen if y != x]
                        for s in set(affected):
                            old_c = counts[s]
                            new_c = counts[s] + delta[s]
                            old_excess = max(0, old_c - 2)
                            new_excess = max(0, new_c - 2)
                            new_viol += new_excess - old_excess
                        
                        if new_viol < 0:  # Reduces violations
                            counts = counts + delta
                            chosen_set.discard(x)
                            chosen.pop(idx)
                            current_violations += new_viol
                            current_size -= 1
                            if current_violations == 0 and current_size > best_size:
                                best_size = current_size
                                best_set = list(chosen)
        
        return best_set
    
    # Phase 1: Greedy initialization
    initial = _greedy_init()
    best_result = list(initial)
    best_size = len(initial)
    
    # Phase 2: Simulated Annealing
    if time.time() < deadline - 10:
        sa_result = _simulated_annealing(initial, deadline - 10)
        if len(sa_result) > best_size:
            best_result = sa_result
            best_size = len(sa_result)
    
    # Phase 3: Local improvement on best result (try adding more)
    def _local_add_improve(chosen, time_limit):
        """Try to add more elements to a feasible set."""
        chosen = list(chosen)
        counts = _build_counts_array(chosen)
        used = set(chosen)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try all unused numbers
            for x in range(1, N + 1):
                if x in used:
                    continue
                # Check if x can be added
                ok = True
                if counts[2 * x] >= 2:
                    ok = False
                else:
                    for y in chosen:
                        if counts[x + y] >= 2:
                            ok = False
                            break
                if ok:
                    chosen.append(x)
                    used.add(x)
                    counts[2 * x] += 1
                    for y in chosen[:-1]:
                        counts[x + y] += 1
                    improved = True
                    break
        return chosen
    
    if best_result and time.time() < deadline - 5:
        improved = _local_add_improve(best_result, deadline - 5)
        if len(improved) > best_size:
            best_result = improved
            best_size = len(improved)
    
    # Phase 4: Multiple SA restarts with different initializations
    for restart in range(3):
        if time.time() > deadline - 15:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Descending greedy
            chosen = []
            counts = [0] * (MAX_SUM + 1)
            for x in range(N, 0, -1):
                ok = True
                if counts[2 * x] >= 2:
                    ok = False
                else:
                    for y in chosen:
                        if counts[x + y] >= 2:
                            ok = False
                            break
                if ok:
                    chosen.append(x)
                    counts[2 * x] += 1
                    for y in chosen[:-1]:
                        counts[x + y] += 1
            init_set = chosen
        elif restart == 1:
            # Random order greedy
            order = list(range(1, N + 1))
            random.shuffle(order)
            chosen = []
            counts = [0] * (MAX_SUM + 1)
            for x in order:
                ok = True
                if counts[2 * x] >= 2:
                    ok = False
                else:
                    for y in chosen:
                        if counts[x + y] >= 2:
                            ok = False
                            break
                if ok:
                    chosen.append(x)
                    counts[2 * x] += 1
                    for y in chosen[:-1]:
                        counts[x + y] += 1
            init_set = chosen
        else:
            # Sparse initialization: start with multiples of a random base
            base = random.randint(3, 10)
            init_set = [x for x in range(1, N + 1) if x % base == 0]
            # Remove violations
            counts = _build_counts_array(init_set)
            while _compute_violations(counts) > 0:
                # Find worst violation
                excess = counts - 2
                worst_sum = int(np.argmax(excess))
                if excess[worst_sum] <= 0:
                    break
                # Find elements that sum to worst_sum and remove one
                removed = False
                for i in range(len(init_set)):
                    for j in range(i, len(init_set)):
                        if init_set[i] + init_set[j] == worst_sum:
                            # Remove the larger one
                            init_set.pop(j if init_set[j] > init_set[i] else i)
                            counts = _build_counts_array(init_set)
                            removed = True
                            break
                    if removed:
                        break
                if not removed:
                    break
        
        if time.time() < deadline - 10:
            sa_res = _simulated_annealing(init_set, deadline - 10)
            if len(sa_res) > best_size:
                best_result = sa_res
                best_size = len(sa_res)
        
        # Local improvement
        if best_result and time.time() < deadline - 5:
            improved = _local_add_improve(best_result, deadline - 5)
            if len(improved) > best_size:
                best_result = improved
                best_size = len(improved)
    
    # Final validation
    def _validate(result):
        if not result:
            return 0
        if len(set(result)) != len(result):
            return 0
        if any(x < 1 or x > N for x in result):
            return 0
        counts = _build_counts_array(result)
        for s in range(2, MAX_SUM + 1):
            if counts[s] > 2:
                return 0
        return len(result)
    
    final_size = _validate(best_result)
    if final_size == 0:
        # Fallback to greedy
        return initial
    
    return best_result
# EVOLVE-BLOCK-END
