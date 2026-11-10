import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    def check_valid(P, P_next):
        # P is list of prefix sums
        # P_next is the new prefix sum
        # Check if P_next - P[k] is in used_diffs
        # We need to pass used_diffs or recompute?
        # Recomputing is O(n^2) per check, which is O(n^3) per step.
        # Better to maintain used_diffs.
        pass

    # State: (P_list, used_diffs_set, last_d)
    # We can just store P_list and recompute used_diffs if needed, or store it.
    # Storing it is better.
    
    def expand(state):
        P, used_diffs, last_d = state
        candidates = []
        for d in range(last_d + 1, 257):
            P_next = P[-1] + d
            valid = True
            new_diffs = []
            for k in range(len(P)):
                diff = P_next - P[k]
                if diff in used_diffs:
                    valid = False
                    break
                new_diffs.append(diff)
            if valid:
                new_used = used_diffs | set(new_diffs)
                new_P = P + [P_next]
                candidates.append((new_P, new_used, d))
        return candidates

    # Beam Search
    beam = [([0], set(), 0)]
    best_seq = []
    
    while beam and time.time() < deadline:
        next_beam = []
        for state in beam:
            candidates = expand(state)
            if candidates:
                next_beam.extend(candidates)
            else:
                # Dead end, evaluate
                P = state[0]
                a = [P[i] - P[i-1] for i in range(1, len(P))]
                if len(a) > len(best_seq):
                    best_seq = a
        
        if not next_beam:
            break
            
        # Prune beam: keep top B
        # Heuristic: prefer states with smaller last_d? Or larger length?
        # All states in the same level have the same length.
        # Smaller last_d leaves more room for future differences.
        # So sort by last_d (ascending) and keep top B.
        B = 50
        next_beam.sort(key=lambda x: x[2])
        beam = next_beam[:B]
        
    # Check final beam states
    for state in beam:
        P = state[0]
        a = [P[i] - P[i-1] for i in range(1, len(P))]
        if len(a) > len(best_seq):
            best_seq = a
            
    return best_seq
