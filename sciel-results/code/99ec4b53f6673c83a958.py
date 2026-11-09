import time
import random
import numpy as np

N = 51975

def get_divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)

DIVS = [d for d in get_divisors(N) if d > 1]

def score_solution(classes):
    covered = bytearray(N)
    for m, r in classes:
        for x in range(r, N, m):
            covered[x] = 1
    return N - sum(covered)

def greedy_construct(divs, N, rng=None):
    if rng is None:
        rng = random
    covered = bytearray(N)
    used = set()
    classes = []
    # Heuristic: Start with larger moduli? Or smaller?
    # Let's try sorting by N/m (density) descending (small m first) or ascending (large m first).
    # Randomize the order for diversity in restarts.
    order = divs[:]
    rng.shuffle(order)
    
    for m in order:
        if m in used:
            continue
        best_r = -1
        best_gain = 0
        # Optimization: If N/m is small, it's cheap to check.
        # We check all r in [0, m)
        # To speed up, we can sample if m is large, but m <= 17325, so full check is fine.
        for r in range(m):
            gain = 0
            # Count uncovered residues in this class
            # Using slicing might be faster in Python?
            # covered[r::m] creates a view/copy.
            # Let's use a loop for clarity, numpy might be faster for counting.
            # But we need to update covered later.
            # Let's stick to Python loops for update, but maybe numpy for count?
            # Actually, just a simple loop is fine for 52k size.
            
            # Faster check:
            # if covered[r::m].count(0) > best_gain:
            #   best_gain = ...
            #   best_r = r
            
            # Let's use the slicing trick for counting, it's C-speed in Python
            cnt = covered[r::m].count(0)
            if cnt > best_gain:
                best_gain = cnt
                best_r = r
                
        if best_gain > 0:
            classes.append([m, best_r])
            used.add(m)
            # Update covered
            for x in range(best_r, N, m):
                covered[x] = 1
    return classes

def local_search(classes, N, max_passes=10):
    if not classes:
        return classes, score_solution(classes)
    
    current_score = score_solution(classes)
    
    for _ in range(max_passes):
        improved = False
        # Try optimizing each class
        for i in range(len(classes)):
            m, r_old = classes[i]
            
            # Calculate coverage of all OTHER classes
            cov_rest = bytearray(N)
            for j in range(len(classes)):
                if j == i:
                    continue
                mj, rj = classes[j]
                for x in range(rj, N, mj):
                    cov_rest[x] = 1
            
            # Find best r for m given cov_rest
            # We want to maximize the number of residues in the class that are 0 in cov_rest
            best_r = r_old
            best_gain = -1
            
            # Tally counts of 0s in cov_rest for each residue mod m
            counts = [0] * m
            # Iterate through all x in [0, N)
            # If cov_rest[x] is 0, increment counts[x % m]
            # This is O(N) per class optimization.
            # With |classes| ~ 20, this is 20 * 52k = 1M ops per pass.
            # 10 passes = 10M ops. Very fast.
            
            for x in range(N):
                if not cov_rest[x]:
                    counts[x % m] += 1
            
            # Find max count
            max_count = max(counts)
            if max_count > -1: # always true
                # Find the r with max count
                # If multiple, pick one (e.g., first)
                best_r = counts.index(max_count)
                
                # Calculate new score contribution
                # New score = (Total Uncovered in cov_rest) - (Covered by new class)
                # Total Uncovered in cov_rest = N - sum(cov_rest)
                # But we don't need absolute score, just improvement.
                # If best_r != r_old, we should check if it actually improves the global score.
                # The gain is max_count.
                # The old gain was counts[r_old].
                if max_count > counts[r_old]:
                    classes[i][1] = best_r
                    improved = True
                    # Update current_score? 
                    # current_score -= (max_count - counts[r_old])
                    
        if not improved:
            break
            
    final_score = score_solution(classes)
    return classes, final_score

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 110s safety margin
    
    best_classes = []
    best_score = N # Max possible score
    
    # Initial greedy run
    divs = DIVS
    classes = greedy_construct(divs, N)
    classes, score = local_search(classes, N)
    if score < best_score:
        best_score = score
        best_classes = classes
    
    # Randomized restarts
    while time.time() < deadline:
        # Randomize the order of divisors in greedy
        classes = greedy_construct(divs, N)
        classes, score = local_search(classes, N, max_passes=5) # Fewer passes for more restarts?
        
        if score < best_score:
            best_score = score
            best_classes = classes
            if best_score == 0:
                break
                
    return best_classes
