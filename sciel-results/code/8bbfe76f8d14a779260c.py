"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin within 60s limit
    
    n = 256
    weights = [Fraction(1, i) for i in range(n + 1)]  # 1-indexed
    
    # Precompute: for each x, which pairs (a,c) make (a,x,c) a 3-AP (x is middle)
    # and which pairs (other_end, middle) make (x, middle, other_end) or (other_end, middle, x) a 3-AP
    middle_pairs = [[] for _ in range(n + 1)]
    end_pairs = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for c in range(a + 2, n + 1, 2):
            b = (a + c) // 2
            if a < b < c:
                middle_pairs[b].append((a, c))
                end_pairs[a].append((c, b))
                end_pairs[c].append((a, b))
    
    def is_valid_add(x, selected):
        """Check if x can be added to selected without forming a 3-AP."""
        for (a, c) in middle_pairs[x]:
            if a in selected and c in selected:
                return False
        for (other_end, middle) in end_pairs[x]:
            if other_end in selected and middle in selected:
                return False
        return True
    
    def compute_score(selected):
        return sum(weights[x] for x in selected)
    
    # Greedy initialization
    selected = set()
    for x in range(1, n + 1):
        if is_valid_add(x, selected):
            selected.add(x)
    
    best_score = compute_score(selected)
    best_set = set(selected)
    
    # Local search
    def local_search(current_set, current_score, deadline):
        nonlocal best_score, best_set
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_list = sorted(current_set)
            
            # Try single removal + addition
            for x in current_list:
                trial = set(current_set)
                trial.discard(x)
                trial_score = current_score - weights[x]
                
                best_y = None
                best_cand = trial_score
                for y in range(1, n + 1):
                    if y in trial:
                        continue
                    if is_valid_add(y, trial):
                        cand = trial_score + weights[y]
                        if cand > best_cand:
                            best_cand = cand
                            best_y = y
                
                if best_y is not None and best_cand > current_score:
                    current_set = trial | {best_y}
                    current_score = best_cand
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(current_set)
                    break
            
            # Try 2-for-1 if single swap didn't help
            if not improved and time.time() < deadline:
                for i in range(len(current_list)):
                    for j in range(i + 1, len(current_list)):
                        x1, x2 = current_list[i], current_list[j]
                        trial = set(current_set)
                        trial.discard(x1)
                        trial.discard(x2)
                        trial_score = current_score - weights[x1] - weights[x2]
                        
                        best_y = None
                        best_cand = trial_score
                        for y in range(1, n + 1):
                            if y in trial:
                                continue
                            if is_valid_add(y, trial):
                                cand = trial_score + weights[y]
                                if cand > best_cand:
                                    best_cand = cand
                                    best_y = y
                        
                        if best_y is not None and best_cand > current_score:
                            current_set = trial | {best_y}
                            current_score = best_cand
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = set(current_set)
                            break
                    if improved:
                        break
            
            # Try 1-for-2
            if not improved and time.time() < deadline:
                for x in current_list:
                    trial = set(current_set)
                    trial.discard(x)
                    trial_score = current_score - weights[x]
                    
                    added = []
                    best_cand = trial_score
                    for y in range(1, n + 1):
                        if y in trial or y in added:
                            continue
                        if is_valid_add(y, trial | set(added)):
                            added.append(y)
                            best_cand += weights[y]
                            if len(added) == 2:
                                break
                    
                    if len(added) == 2 and best_cand > current_score:
                        current_set = trial | set(added)
                        current_score = best_cand
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = set(current_set)
                        break
        
        return current_set, current_score
    
    selected, best_score = local_search(selected, best_score, deadline)
    
    return list(best_set)
# EVOLVE-BLOCK-END
