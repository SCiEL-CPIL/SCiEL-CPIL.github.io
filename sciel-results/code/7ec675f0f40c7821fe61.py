import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
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

    def greedy_extend(P, used_diffs, last_d, time_limit):
        """Greedily extend the sequence as far as possible."""
        while time.time() < time_limit:
            found = False
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
                    P = P + [P_next]
                    used_diffs = used_diffs | set(new_diffs)
                    last_d = d
                    found = True
                    break
            if not found:
                break
        return P, used_diffs, last_d

    # Beam Search with improved heuristic
    beam = [([0], set(), 0)]
    best_seq = []
    best_len = 0
    
    # Try multiple beam widths and pick the best
    for B in [30, 50, 80]:
        if time.time() > deadline - 5:
            break
            
        beam = [([0], set(), 0)]
        local_best = []
        
        while beam and time.time() < deadline - 10:
            next_beam = []
            for state in beam:
                candidates = expand(state)
                if candidates:
                    next_beam.extend(candidates)
                else:
                    P = state[0]
                    a = [P[i] - P[i-1] for i in range(1, len(P))]
                    if len(a) > len(local_best):
                        local_best = a
            
            if not next_beam:
                break
                
            # Heuristic: prefer states with smaller last_d (more room) 
            # and smaller total prefix sum (more headroom)
            def state_score(s):
                P, used_diffs, last_d = s
                # Smaller last_d is better (more choices for next step)
                # Smaller max prefix sum is better (more room in [1,256])
                # Fewer used diffs is better (more flexibility)
                return (last_d, P[-1], len(used_diffs))
            
            next_beam.sort(key=state_score)
            beam = next_beam[:B]
            
            # Track best from current beam
            for state in beam:
                P = state[0]
                a = [P[i] - P[i-1] for i in range(1, len(P))]
                if len(a) > len(local_best):
                    local_best = a
        
        # Extend the best beam states greedily
        if beam and time.time() < deadline - 3:
            # Sort beam by heuristic and try extending top few
            def state_score(s):
                P, used_diffs, last_d = s
                return (last_d, P[-1])
            
            beam.sort(key=state_score)
            for state in beam[:min(5, len(beam))]:
                if time.time() > deadline - 2:
                    break
                P, used_diffs, last_d = greedy_extend(state[0], state[1], state[2], deadline - 1)
                a = [P[i] - P[i-1] for i in range(1, len(P))]
                if len(a) > len(local_best):
                    local_best = a
        
        if len(local_best) > best_len:
            best_seq = local_best
            best_len = len(local_best)
    
    # Final check on any remaining beam states
    if beam and time.time() < deadline:
        for state in beam:
            P = state[0]
            a = [P[i] - P[i-1] for i in range(1, len(P))]
            if len(a) > best_len:
                best_seq = a
                best_len = len(a)
    
    return best_seq
