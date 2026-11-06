import time
import random
import sys

N = 18

# Precompute target masks for bitwise verification
# Input k (0 to 2^N - 1) has popcount(k) ones.
# Sorted output has popcount(k) ones at the end (highest indices).
# Wire i is 1 in output iff i >= N - popcount(k).
# Mask T[i] has bit k set to 1 if i >= N - popcount(k).

def precompute_targets():
    targets = [0] * N
    for k in range(1 << N):
        pc = bin(k).count('1')
        # Ones are at indices N-pc to N-1
        start = N - pc
        for i in range(start, N):
            targets[i] |= (1 << k)
    return targets

TARGETS = precompute_targets()

def verify_network(net):
    # State: list of N bitmasks
    # Initial state: wire i has bit k set if input k has bit i set.
    # Input k is an integer. Bit i of k is (k >> i) & 1.
    # So initial mask for wire i is the bit i of all k's.
    # Mask for wire i: bit k is 1 iff (k >> i) & 1 == 1.
    # This is a pattern.
    # We can compute it or just initialize from scratch.
    # Faster:
    # M[i] = 0
    # for k in range(1<<N):
    #   for i in range(N):
    #     if (k>>i)&1: M[i] |= (1<<k)
    
    # Optimization:
    # M[i] is the i-th bit of the counter 0..2^N-1.
    # We can construct M[i] efficiently.
    M = [0] * N
    # For each wire i, the bit is 1 for half the inputs.
    # Pattern: 0...0 1...1 0...0 1...1
    # Block size 2^i.
    # Actually, let's just use the loop for simplicity if it's fast enough.
    # 2^18 * 18 = 4.7M ops. Might be slow in Python if done in a loop.
    
    # Better initialization:
    # M[i] = sum(1 << k for k in range(1<<N) if (k>>i)&1)
    # This is still O(2^N).
    
    # Let's use a recursive or bitwise construction.
    # M[0] = 0x5555... (alternating bits)
    # M[1] = 0x3333...
    # M[i] is a mask with 2^(i-1) zeros followed by 2^(i-1) ones, repeated.
    
    for i in range(N):
        block = 1 << i
        pattern = (1 << block) - 1
        # Repeat pattern
        m = 0
        for start in range(0, 1 << N, block * 2):
            m |= pattern << start
        M[i] = m

    for i, j in net:
        # Compare-exchange: min to i, max to j
        # min(a,b) = a & b
        # max(a,b) = a | b
        mi = M[i]
        mj = M[j]
        M[i] = mi & mj
        M[j] = mi | mj
        
    # Check against targets
    for i in range(N):
        if M[i] != TARGETS[i]:
            return False
    return True

def construct():
    # Seed: Odd-even transposition
    net = [(i, i + 1) for phase in range(N)
           for i in range(phase % 2, N - 1, 2)]
    
    best_net = list(net)
    best_count = len(best_net)
    
    start_time = time.time()
    timeout = 85.0 # 5s margin
    
    # 1. Greedy Removal
    changed = True
    while changed and time.time() - start_time < timeout:
        changed = False
        for idx in range(len(best_net) - 1, -1, -1):
            # Try removing best_net[idx]
            candidate = best_net[:idx] + best_net[idx+1:]
            if verify_network(candidate):
                best_net = candidate
                best_count = len(best_net)
                changed = True
                # Restart loop as indices shifted
                break
                
    # 2. Local Search (Hill Climbing with Random Jumps)
    # We try to find a network with fewer comparators.
    # Moves:
    # - Remove a random comparator
    # - Add a random comparator
    # - Swap two comparators
    
    current_net = list(best_net)
    current_count = len(current_net)
    
    # Pre-generate random pairs for adding
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() - start_time < timeout:
        # Pick a random move
        move_type = random.randint(0, 2)
        
        if move_type == 0 and len(current_net) > 0:
            # Remove random
            idx = random.randint(0, len(current_net) - 1)
            candidate = current_net[:idx] + current_net[idx+1:]
        elif move_type == 1:
            # Add random
            # Avoid duplicates? Not strictly necessary for correctness, but redundant.
            # We can just add any pair.
            pair = random.choice(all_pairs)
            candidate = current_net + [pair]
        else:
            # Swap two
            if len(current_net) < 2:
                continue
            i, j = random.sample(range(len(current_net)), 2)
            candidate = current_net[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            
        if verify_network(candidate):
            if len(candidate) < best_count:
                best_net = candidate
                best_count = len(best_net)
                current_net = candidate
                current_count = len(current_net)
            else:
                # Accept sideways moves to escape local minima?
                # If we accept a move that doesn't improve, we might wander.
                # But if we accept a move that increases count, we are moving away from the goal.
                # However, it might lead to a better basin.
                # Let's accept with a small probability if it doesn't increase count too much?
                # Or just accept if it's the same count (swap) to shuffle.
                if len(candidate) <= current_count:
                    current_net = candidate
                    current_count = len(current_net)
        else:
            # If invalid, reject
            pass

    return best_net

# EVOLVE-BLOCK-START
def construct():
    # Odd-even transposition sorting network: 120 comparators for 16 wires.
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
# EVOLVE-BLOCK-END
