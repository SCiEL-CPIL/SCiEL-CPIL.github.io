import time
import numpy as np
import random

N = 18
TIME_LIMIT = 55  # Safety margin for 90s limit (60s eval + buffer)

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Pre-compute bit-parallel states for fast verification
    # 2^18 inputs, packed 64 per uint64
    num_blocks = 2**N // 64  # 4096 blocks
    
    # states[w][block] bit b is 1 if input (block*64+b) has 1 on wire w
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    # Vectorized initialization: for each wire w, compute the 64-bit mask for each block
    # Wire w of input k is (k >> w) & 1
    # For a block starting at base = b*64, the 64 inputs are base, base+1, ..., base+63
    for w in range(N):
        if w < 6:
            # Period is 2^(w+1), half-period is 2^w
            # For each block, compute the mask
            half = 1 << w
            period = 1 << (w + 1)
            # For block b, base = b * 64
            # The w-th bit of (base + x) for x in 0..63
            # base % period determines the phase
            # Since 64 = 2^6 and period = 2^(w+1) where w < 6, period divides 64
            # So base % period = (b * 64) % period
            # Actually, since period divides 64, (b*64) % period = 0 for all b
            # Wait, that's not right. Let me think again.
            # 64 = 2^6. period = 2^(w+1) for w in 0..5, so period is 2,4,8,16,32
            # All divide 64, so (b*64) % period = 0
            # So the pattern always starts at phase 0
            # The mask is: first half bits are 0, next half bits are 1, repeating
            # Within 64 bits: 64/period repeats of the pattern
            # Pattern: 0...0 1...1 (half zeros, half ones)
            # For w=0: pattern 010101... (32 repeats of 01)
            # For w=1: pattern 00110011... (16 repeats of 0011)
            # etc.
            # Build the mask
            mask = 0
            for rep in range(64 // period):
                for bit in range(half):
                    mask |= (1 << (rep * period + half + bit))
            states[w].fill(mask)
        else:
            # w >= 6: the w-th bit is constant across each block of 64
            # For block b, base = b * 64, bit w of base is (b * 64 >> w) & 1 = (b >> (w-6)) & 1
            for b in range(num_blocks):
                if (b >> (w - 6)) & 1:
                    states[w][b] = (1 << 64) - 1
                else:
                    states[w][b] = 0

    # Pre-allocate workspace for verification
    workspace = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    def verify(comparators):
        """Verify network sorts all inputs. Uses pre-allocated workspace."""
        for w in range(N):
            workspace[w] = states[w]
        
        for i, j in comparators:
            a = workspace[i]
            b = workspace[j]
            workspace[i] = a & b
            workspace[j] = a | b
        
        for w in range(N - 1):
            if np.any(workspace[w] & ~workspace[w + 1]):
                return False
        return True

    def greedy_reduce(net):
        """Remove redundant comparators using fast verification."""
        cur_net = list(net)
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            idx = 0
            n = len(cur_net)
            while idx < n:
                if time.time() > deadline - 2:
                    return cur_net
                # Try removing comparator at idx
                saved = cur_net[idx]
                del cur_net[idx]
                n -= 1
                if verify(cur_net):
                    changed = True
                    # Keep it removed, don't increment idx
                else:
                    cur_net.insert(idx, saved)
                    n += 1
                    idx += 1
        return cur_net

    # Initial Network
    best_network = _odd_even_transposition()
    best_network = greedy_reduce(best_network)
    best_len = len(best_network)
    
    # Local Search
    current_network = list(best_network)
    stagnation_counter = 0
    max_stagnation = 150
    
    # Pre-generate some useful comparator candidates
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() < deadline - 5:
        # Choose a move - bias toward size-reducing moves
        move_type = random.random()
        
        if move_type < 0.30:
            # Mutation: Replace a random comparator with a new one
            if len(current_network) > 0:
                idx = random.randint(0, len(current_network) - 1)
                a, b = random.sample(range(N), 2)
                if a > b: a, b = b, a
                trial = list(current_network)
                trial[idx] = (a, b)
                
        elif move_type < 0.45:
            # Swap: Swap adjacent comparators
            if len(current_network) > 1:
                idx = random.randint(0, len(current_network) - 2)
                trial = list(current_network)
                trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
                
        elif move_type < 0.60:
            # Insert: Add a random comparator
            idx = random.randint(0, len(current_network))
            a, b = random.sample(range(N), 2)
            if a > b: a, b = b, a
            trial = current_network[:idx] + [(a, b)] + current_network[idx:]
            
        elif move_type < 0.75:
            # Move: Relocate a comparator to a new position
            if len(current_network) > 1:
                idx = random.randint(0, len(current_network) - 1)
                comp = current_network[idx]
                new_idx = random.randint(0, len(current_network) - 1)
                trial = list(current_network)
                del trial[idx]
                if new_idx > idx:
                    new_idx -= 1
                trial.insert(new_idx, comp)
        else:
            # Delete: Remove a random comparator (aggressive size reduction)
            if len(current_network) > 1:
                idx = random.randint(0, len(current_network) - 1)
                trial = current_network[:idx] + current_network[idx+1:]
        
        # Apply move and reduce
        reduced = greedy_reduce(trial)
        
        # Update best
        if len(reduced) < len(best_network):
            best_network = reduced
            best_len = len(reduced)
            current_network = reduced
            stagnation_counter = 0
        elif len(reduced) < len(current_network):
            current_network = reduced
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
            if stagnation_counter >= max_stagnation:
                # Kick: Perturb the best network
                K = random.randint(2, 5)
                current_network = list(best_network)
                for _ in range(K):
                    if not current_network: break
                    idx = random.randint(0, len(current_network) - 1)
                    a, b = random.sample(range(N), 2)
                    if a > b: a, b = b, a
                    current_network[idx] = (a, b)
                
                current_network = greedy_reduce(current_network)
                stagnation_counter = 0

    return best_network
