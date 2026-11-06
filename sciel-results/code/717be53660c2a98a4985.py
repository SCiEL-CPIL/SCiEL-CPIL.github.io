import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_sequence = []
    best_len = 0
    
    # Use a list to hold the best sequence so it can be updated in nested scope
    # Or just use nonlocal if we define nested functions properly.
    # But we have multiple restarts, so we can just keep local vars in construct 
    # and pass them to a helper that returns the best found.
    
    def expand_beam(beam, width):
        candidates = []
        
        for seq, sums_set in beam:
            last_val = seq[-1] if seq else 0
            
            # Pre-compute suffix sums of seq: [0, a_k, a_{k-1}+a_k, ...]
            suffix_sums = [0]
            s = 0
            for a in reversed(seq):
                s += a
                suffix_sums.append(s)
                
            # Try all x > last_val
            for x in range(last_val + 1, 257):
                valid = True
                new_sums_list = []
                for suf in suffix_sums:
                    val = x + suf
                    if val in sums_set:
                        valid = False
                        break
                    new_sums_list.append(val)
                
                if valid:
                    new_sums_set = sums_set | set(new_sums_list)
                    candidates.append((seq + [x], new_sums_set))
        
        if not candidates:
            return []
        
        # Diversify selection
        random.shuffle(candidates)
        return candidates[:width]

    def run_beam_search(width=100, max_steps=40):
        current_beam = [([], set())]
        local_best_seq = []
        local_best_len = 0
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            current_beam = expand_beam(current_beam, width)
            
            if not current_beam:
                break
            
            # All sequences in current_beam have length step + 1
            if step + 1 > local_best_len:
                local_best_len = step + 1
                # Pick a random one from the beam to represent this length
                # This ensures we don't always pick the same "best" structure
                local_best_seq = random.choice(current_beam)[0]

        return local_best_seq

    num_restarts = 60
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(2000 + i * 7)
        
        # Vary parameters for diversity
        width = 80 + (i % 5) * 10  # 80, 90, 100, 110, 120
        max_steps = 35 + (i % 3) * 5 # 35, 40, 45
        
        seq = run_beam_search(width, max_steps)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_sequence = seq

    return best_sequence
