import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s buffer
    
    def compute_block_sums(seq):
        """Compute all consecutive block sums. Returns (sums_set, sum_count_dict)."""
        n = len(seq)
        sums = set()
        count_dict = defaultdict(int)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
                count_dict[s] += 1
        return sums, count_dict
    
    def count_duplicates(seq):
        """Count the number of duplicate block sums (soft objective for SA)."""
        n = len(seq)
        count_dict = defaultdict(int)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                count_dict[s] += 1
        dup_count = 0
        for c in count_dict.values():
            if c > 1:
                dup_count += c - 1
        return dup_count
    
    def is_valid(seq):
        """Check if all consecutive block sums are distinct."""
        n = len(seq)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def try_extend_dfs(seq, sums_set, max_depth, deadline):
        """Try to extend seq using DFS. Returns best extension found."""
        best = seq[:]
        best_len = len(seq)
        
        def dfs(current_seq, current_sums, depth):
            nonlocal best, best_len
            if time.time() > deadline:
                return
            if len(current_seq) > best_len:
                best_len = len(current_seq)
                best = current_seq[:]
            if depth >= max_depth:
                return
            last_val = current_seq[-1] if current_seq else 0
            for x in range(last_val + 1, 257):
                if time.time() > deadline:
                    return
                # Check if x can be appended
                new_sums = []
                s = x
                conflict = False
                for ns in new_sums:
                    pass
                # Compute new suffix sums
                suffix = [x]
                total = x
                for a in reversed(current_seq):
                    total += a
                    suffix.append(total)
                for ns in suffix:
                    if ns in current_sums:
                        conflict = True
                        break
                if conflict:
                    continue
                new_sums_set = current_sums.union(suffix)
                dfs(current_seq + [x], new_sums_set, depth + 1)
        
        dfs(seq, sums_set, 0)
        return best
    
    # Phase 1: Simulated Annealing on difference vectors
    # Model: sequence a[0..n-1], differences d[i] = a[i+1] - a[i] >= 1
    # Objective: minimize number of duplicate block sums
    
    best_seq = []
    best_len = 0
    best_dup_count = float('inf')
    
    # Start with a greedy baseline
    greedy_seq = []
    greedy_sums = set()
    for x in range(1, 257):
        suffix = [x]
        total = x
        for a in reversed(greedy_seq):
            total += a
            suffix.append(total)
        if not any(s in greedy_sums for s in suffix):
            greedy_seq.append(x)
            greedy_sums.update(suffix)
    
    best_seq = greedy_seq[:]
    best_len = len(greedy_seq)
    best_dup_count = 0
    
    # SA parameters
    sa_start_temp = 5.0
    sa_end_temp = 0.01
    sa_iterations = 3000
    sa_max_len = 16  # Try sequences up to length 16
    
    # Initialize SA with a random valid sequence of length ~8
    def random_valid_seq(target_len):
        """Generate a random strictly increasing sequence in [1,256] of given length."""
        if target_len <= 0:
            return []
        # Choose random values
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        # Try to build a valid sequence
        for _ in range(100):
            seq = sorted(random.sample(candidates, target_len))
            if is_valid(seq):
                return seq
        return []
    
    # Run SA for various target lengths
    for target_len in range(6, sa_max_len + 1):
        if time.time() > deadline - 10:
            break
        
        # Initialize
        current_seq = random_valid_seq(target_len)
        if not current_seq:
            # Fallback: use a simple arithmetic progression
            current_seq = list(range(1, target_len + 1))
        
        current_dup = count_duplicates(current_seq)
        if current_dup == 0 and len(current_seq) > best_len:
            best_seq = current_seq[:]
            best_len = len(current_seq)
            best_dup_count = 0
        
        best_sa_seq = current_seq[:]
        best_sa_dup = current_dup
        
        # SA loop
        for it in range(sa_iterations):
            if time.time() > deadline - 10:
                break
            
            temp = sa_start_temp * (sa_end_temp / sa_start_temp) ** (it / sa_iterations)
            
            # Perturb: change one difference
            new_seq = current_seq[:]
            if len(new_seq) >= 2:
                pos = random.randint(0, len(new_seq) - 2)
                old_diff = new_seq[pos + 1] - new_seq[pos]
                # Try changing the difference
                delta = random.choice([-1, -2, 1, 2, 3, -3])
                new_diff = old_diff + delta
                if new_diff >= 1:
                    # Adjust subsequent elements to maintain strictly increasing
                    new_val = new_seq[pos] + new_diff
                    if new_val < new_seq[pos + 1] if pos + 1 < len(new_seq) else True:
                        if new_val <= 256 and (pos == 0 or new_val > new_seq[pos - 1]):
                            new_seq[pos + 1] = new_val
                            # Propagate to maintain increasing
                            for k in range(pos + 2, len(new_seq)):
                                if new_seq[k] <= new_seq[k - 1]:
                                    new_seq[k] = new_seq[k - 1] + 1
                                if new_seq[k] > 256:
                                    break
                            if all(1 <= v <= 256 for v in new_seq) and all(new_seq[i] < new_seq[i+1] for i in range(len(new_seq)-1)):
                                new_dup = count_duplicates(new_seq)
                                delta_dup = new_dup - current_dup
                                if delta_dup < 0 or random.random() < pow(2.718281828, -delta_dup / max(temp, 0.01)):
                                    current_seq = new_seq
                                    current_dup = new_dup
                                    if current_dup < best_sa_dup:
                                        best_sa_dup = current_dup
                                        best_sa_seq = current_seq[:]
                                    if current_dup == 0 and len(current_seq) > best_len:
                                        best_seq = current_seq[:]
                                        best_len = len(current_seq)
                                        best_dup_count = 0
            else:
                # Single element, just change its value
                new_val = random.randint(1, 256)
                new_seq = [new_val]
                new_dup = count_duplicates(new_seq)
                delta_dup = new_dup - current_dup
                if delta_dup < 0 or random.random() < pow(2.718281828, -delta_dup / max(temp, 0.01)):
                    current_seq = new_seq
                    current_dup = new_dup
        
        # After SA, try to extend the best found
        if best_sa_dup == 0 and len(best_sa_seq) > best_len:
            best_seq = best_sa_seq[:]
            best_len = len(best_sa_seq)
    
    # Phase 2: DFS extension from best SA result
    if time.time() < deadline - 5:
        # Get sums for best_seq
        bs, _ = compute_block_sums(best_seq)
        extended = try_extend_dfs(best_seq, bs, max_depth=4, deadline=deadline - 2)
        if len(extended) > best_len:
            best_seq = extended
            best_len = len(best_seq)
    
    # Phase 3: Final local search - try inserting elements
    if time.time() < deadline - 2 and len(best_seq) < 18:
        current = best_seq[:]
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(current) + 1):
                if time.time() > deadline - 2:
                    break
                prev_val = current[i-1] if i > 0 else 0
                next_val = current[i] if i < len(current) else 257
                for x in range(prev_val + 1, min(next_val, 257)):
                    test_seq = current[:i] + [x] + current[i:]
                    if is_valid(test_seq):
                        current = test_seq
                        best_seq = current[:]
                        best_len = len(best_seq)
                        improved = True
                        break
                if improved:
                    break
    
    # Ensure result is valid
    if not is_valid(best_seq):
        # Fallback to greedy
        greedy_seq = []
        greedy_sums = set()
        for x in range(1, 257):
            suffix = [x]
            total = x
            for a in reversed(greedy_seq):
                total += a
                suffix.append(total)
            if not any(s in greedy_sums for s in suffix):
                greedy_seq.append(x)
                greedy_sums.update(suffix)
        best_seq = greedy_seq
    
    return best_seq
