import time
import sys

def construct():
    start_time = time.time()
    time_limit = 58.0  # Safety margin for 60s limit
    
    best_seq = []
    best_len = 0
    
    def _greedy_construct():
        """Quick greedy: at each step, pick the smallest valid x."""
        seq = []
        suffix_sums = []
        seen_sums = set()
        last_val = 0
        
        while True:
            if time.time() - start_time > time_limit:
                break
            found = False
            for x in range(last_val + 1, 257):
                valid = True
                new_sums = []
                for s in suffix_sums:
                    ns = x + s
                    if ns in seen_sums:
                        valid = False
                        break
                    new_sums.append(ns)
                if not valid:
                    continue
                if x in seen_sums:
                    continue
                new_sums.append(x)
                
                seq.append(x)
                suffix_sums = [x + s for s in suffix_sums] + [x]
                for ns in new_sums:
                    seen_sums.add(ns)
                last_val = x
                found = True
                break
            if not found:
                break
        
        return seq
    
    def dfs(seq, suffix_sums, seen_sums, last_val):
        nonlocal best_seq, best_len, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)
            
        # Pruning: Can we possibly beat best_len?
        if len(seq) + (256 - last_val) <= best_len:
            return
            
        # Try adding next value x
        for x in range(last_val + 1, 257):
            if time.time() - start_time > time_limit:
                return
                
            new_sums = []
            valid = True
            for s in suffix_sums:
                ns = x + s
                if ns in seen_sums:
                    valid = False
                    break
                new_sums.append(ns)
            
            if not valid:
                continue
                
            if x in seen_sums:
                continue
            new_sums.append(x)
            
            new_suffix_sums = [x + s for s in suffix_sums] + [x]
            
            for ns in new_sums:
                seen_sums.add(ns)
            
            seq.append(x)
            dfs(seq, new_suffix_sums, seen_sums, x)
            seq.pop()
            
            for ns in new_sums:
                seen_sums.discard(ns)
    
    # Step 1: Quick greedy to get a good starting point for pruning
    greedy_seq = _greedy_construct()
    if len(greedy_seq) > best_len:
        best_len = len(greedy_seq)
        best_seq = list(greedy_seq)
    
    # Step 2: DFS to try to beat the greedy solution
    dfs([], [], set(), 0)
    
    if not best_seq:
        best_seq = [1]
        
    return best_seq
