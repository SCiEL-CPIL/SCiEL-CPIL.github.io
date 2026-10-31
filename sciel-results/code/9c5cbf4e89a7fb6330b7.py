import math
import time

def construct():
    start_time = time.time()
    deadline = 100.0  # 100s safety margin
    
    best_set = [0, 1, 3, 7]
    best_score = -1.0
    
    def score_set(u):
        if not u or 0 not in u:
            return -1.0
        u = sorted(set(u))
        max_u = max(u)
        if max_u == 0:
            return 0.0
        sumset = set()
        diffset = set()
        for a in u:
            for b in u:
                sumset.add(a + b)
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 1.0:
            return 0.0
        val = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
        return val
    
    best_score = score_set(best_set)
    
    # Local search
    # Try adding elements
    # Try removing elements
    # Try swapping elements
    
    # Simple hill climbing
    current_set = best_set[:]
    current_score = best_score
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try adding a new element
        max_val = max(current_set)
        for new_val in range(1, max_val * 2 + 10):
            if new_val in current_set:
                continue
            candidate = current_set + [new_val]
            candidate.sort()
            s = score_set(candidate)
            if s > current_score:
                current_set = candidate
                current_score = s
                improved = True
                if s > best_score:
                    best_score = s
                    best_set = current_set[:]
        
        # Try removing an element (keep 0)
        if len(current_set) > 2:
            for i in range(1, len(current_set)):
                candidate = current_set[:i] + current_set[i+1:]
                s = score_set(candidate)
                if s > current_score:
                    current_set = candidate
                    current_score = s
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = current_set[:]
                        
        # Try swapping an element
        for i in range(1, len(current_set)):
            old_val = current_set[i]
            for new_val in range(1, max(current_set) * 2 + 10):
                if new_val == old_val or new_val in current_set:
                    continue
                candidate = current_set[:]
                candidate[i] = new_val
                candidate.sort()
                s = score_set(candidate)
                if s > current_score:
                    current_set = candidate
                    current_score = s
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = current_set[:]
                        
    return best_set
