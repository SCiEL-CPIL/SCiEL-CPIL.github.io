import itertools
import random
import time

N = 16
NUM_POINTS = N * N
POINTS = list(itertools.product(range(N), repeat=2))
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

# Precomputation
# partners[q][p] = list of indices a such that {p, a} is a pair in B(q)
# i.e., (q, p, a) is isosceles.
partners = [[[] for _ in range(NUM_POINTS)] for _ in range(NUM_POINTS)]

# We also need to know for a given set S, the counts.
# But partners is static.

def is_isosceles(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    d12 = (x1-x2)**2 + (y1-y2)**2
    d13 = (x1-x3)**2 + (y1-y3)**2
    d23 = (x2-x3)**2 + (y2-y3)**2
    return d12 == d13 or d12 == d23 or d13 == d23

# Build partners table
# For each q, for each p, find all a such that (q, p, a) is isosceles.
# This is O(N^6) which is 16^6 = 16M. Might be slow in Python if not optimized.
# 16M iterations with simple math is ~1-2 seconds. Acceptable.
start_time = time.time()
for q_idx, q in enumerate(POINTS):
    for p_idx, p in enumerate(POINTS):
        if q_idx == p_idx:
            continue
        for a_idx, a in enumerate(POINTS):
            if a_idx == q_idx or a_idx == p_idx:
                continue
            if is_isosceles(q, p, a):
                partners[q_idx][p_idx].append(a_idx)
# Note: partners[q][p] contains a. The pair is {p, a}.
# When we add p to S, we increment counts[q] by number of a in partners[q][p] that are in S.
# When we remove p from S, we decrement counts[q] by number of a in partners[q][p] that are in S (excluding p itself, which is not in S anymore, but a is).

def construct():
    deadline = time.time() + 290 # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    # Helper to run one instance of the heuristic
    def run_heuristic(seed_val):
        random.seed(seed_val)
        
        # 1. Generate initial maximal independent set
        # We use the counts array to track blocking.
        counts = [0] * NUM_POINTS
        S = set()
        
        # Shuffle points for randomization
        indices = list(range(NUM_POINTS))
        random.shuffle(indices)
        
        for p_idx in indices:
            # Check if p_idx can be added
            # It can be added if counts[p_idx] == 0
            # (i.e., no pair in S blocks p_idx)
            if counts[p_idx] == 0:
                S.add(p_idx)
                # Update counts for all q
                # For each q, counts[q] increases by number of a in partners[q][p_idx] that are in S
                # Note: p_idx is now in S. The pairs are {p_idx, a}.
                # We need to count how many a in partners[q][p_idx] are already in S.
                for q_idx in range(NUM_POINTS):
                    # Optimization: only update if partners[q_idx][p_idx] is not empty
                    if partners[q_idx][p_idx]:
                        # Count a in S
                        # Since S is a set, lookup is O(1)
                        # But iterating over partners list is O(len)
                        cnt = 0
                        for a_idx in partners[q_idx][p_idx]:
                            if a_idx in S:
                                cnt += 1
                        if cnt > 0:
                            counts[q_idx] += cnt
        
        # 2. Reduction phase
        # Try to remove points
        # We iterate until no more removals are possible
        improved = True
        while improved:
            improved = False
            # Shuffle S for randomization in removal order
            s_list = list(S)
            random.shuffle(s_list)
            
            for p_idx in s_list:
                if p_idx not in S:
                    continue
                
                # Check if we can remove p_idx
                # We need to check if S \ {p_idx} is still maximal.
                # This requires checking all q not in S \ {p_idx}.
                # q not in S \ {p_idx} means q not in S OR q == p_idx.
                
                can_remove = True
                # We need to calculate the impact of removing p_idx on counts
                # impact[q] = number of a in partners[q][p_idx] that are in S (excluding p_idx, but p_idx not in partners list as a? 
                # partners[q][p_idx] contains a such that {p_idx, a} is isosceles with q.
                # So a is distinct from p_idx.
                # The pairs are {p_idx, a}.
                # If we remove p_idx, these pairs are no longer in S.
                # So counts[q] decreases by the number of such a that are in S.
                
                # We can compute this impact on the fly
                # If any q not in S' has new_count[q] == 0, we cannot remove.
                
                # To speed up, we can first check q = p_idx.
                # new_count[p_idx] = counts[p_idx] - impact[p_idx]
                # If new_count[p_idx] == 0, we can't remove (because p_idx would be unblocked).
                
                # Let's compute impact for all q? No, only for q not in S'.
                # q not in S' is {q | q not in S} U {p_idx}.
                
                # We will iterate through all q in that set.
                # If N=16, NUM_POINTS=256. This is small.
                
                # Pre-calculate impact for p_idx?
                # impact[q] = sum(1 for a in partners[q][p_idx] if a in S)
                
                # We can just loop q.
                for q_idx in range(NUM_POINTS):
                    if q_idx in S and q_idx != p_idx:
                        continue # q is in S', so we don't care if it's blocked (it's already in the set)
                    
                    # Calculate new count for q
                    # new_count = counts[q_idx] - impact
                    impact = 0
                    for a_idx in partners[q_idx][p_idx]:
                        if a_idx in S:
                            impact += 1
                    
                    new_count = counts[q_idx] - impact
                    if new_count == 0:
                        can_remove = False
                        break
                
                if can_remove:
                    # Perform removal
                    S.remove(p_idx)
                    # Update counts
                    for q_idx in range(NUM_POINTS):
                        if not partners[q_idx][p_idx]:
                            continue
                        impact = 0
                        for a_idx in partners[q_idx][p_idx]:
                            if a_idx in S:
                                impact += 1
                        counts[q_idx] -= impact
                    improved = True
                    break # Restart the while loop to pick up new state
        
        return len(S), S

    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        size, s = run_heuristic(restarts)
        if size < best_size:
            best_size = size
            best_set = s
        restarts += 1
        # If we find a very small set, we can stop early? 
        # Theoretical min is small. Let's just run until time.
        
    if best_set is None:
        # Fallback to seed logic if something went wrong
        chosen = []
        for p in itertools.product(range(N), repeat=2):
            if all(not is_isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
                chosen.append(p)
        return chosen
        
    # Convert indices back to points
    return [POINTS[i] for i in best_set]

if __name__ == "__main__":
    # Quick test
    s = construct()
    print(len(s))
