"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Bitset representation: x in 1..512 maps to bit (x-1)
    # S is an integer where bit i is 1 if (i+1) is in the set
    
    def bitset_from_set(S):
        b = 0
        for x in S:
            b |= (1 << (x - 1))
        return b
    
    def set_from_bitset(b):
        s = []
        i = 0
        while b:
            if b & 1:
                s.append(i + 1)
            b >>= 1
            i += 1
        return s
    
    def score_bitset(b):
        count = 0
        for d in range(1, 257):
            if (b & (b >> d) & (b >> (2 * d))):
                count += 1
        return count
    
    def popcount(b):
        return bin(b).count('1')
    
    def random_set():
        s = set(random.sample(range(1, 513), 32))
        return s
    
    # Start with a few different initial strategies
    best_set = None
    best_score = -1
    
    # Strategy 1: Random restarts with hill climbing
    num_restarts = 20
    max_iters_per_restart = 500
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate initial set
        if restart == 0:
            # Start with a structured set (arithmetic progression-like)
            s = set()
            # Try a few structured starts
            for base in range(1, 513, 16):
                for step in range(1, 16):
                    candidate = set()
                    for i in range(32):
                        val = base + i * step
                        if val <= 512:
                            candidate.add(val)
                    if len(candidate) == 32:
                        s = candidate
                        break
                if len(s) == 32:
                    break
            if len(s) != 32:
                s = random_set()
        else:
            s = random_set()
        
        b = bitset_from_set(s)
        current_score = score_bitset(b)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(s)
        
        # Hill climbing
        for it in range(max_iters_per_restart):
            if time.time() > deadline - 5.0:
                break
            
            # Try swapping one element with another
            # Pick a random element to remove and a random element to add
            removed = random.choice(list(s))
            added = random.randint(1, 512)
            
            if added == removed or added in s:
                continue
            
            new_s = set(s)
            new_s.remove(removed)
            new_s.add(added)
            
            if len(new_s) != 32:
                continue
            
            new_b = bitset_from_set(new_s)
            new_score = score_bitset(new_b)
            
            if new_score > current_score:
                s = new_s
                b = new_b
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(s)
    
    # Final check
    if best_set is None or len(best_set) != 32:
        best_set = set(range(1, 33))
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
