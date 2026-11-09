import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 88.0  # safety margin before 90s kill
    
    N = 128
    
    # Bitmask representation:
    # Value v (1..128) is represented by bit (v-1) in an integer.
    
    # Precompute valid 'a' for each x such that x+a <= 128
    # a must be in 1..128.
    valid_a = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if x + a <= N:
                valid_a[x].append(a)
                
    # Precompute bit masks for sums?
    # sum_bit[a][b] = 1 << (a+b-1) ?
    # This is a 129x129 matrix of ints. 129*129*8 bytes ~ 130KB. Fine.
    sum_bit = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            s = a + b
            if s <= N:
                sum_bit[a][b] = 1 << (s - 1)
            else:
                sum_bit[a][b] = 0

    def can_add(x, S_mask):
        """
        Check if x can be added to S (represented by S_mask) without creating a forbidden triple.
        A forbidden triple {x, a, b} requires:
        1. x, a, b in S' (x is new, a,b in S)
        2. x+a in S' -> x+a in S (since x+a != x, a, b generally, and x is not in S yet)
           Wait, if x+a == a? No, x>=1.
           If x+a == b? Then b is in S, so x+a is in S'.
           If x+a == x? No.
           So we need x+a in S OR x+a == b (which is in S).
           Actually, the condition is "x+a is selected".
           Since S' = S U {x}, "x+a is selected" means x+a in S OR x+a == x.
           x+a == x is impossible for a>=1.
           So we strictly need x+a in S.
           Similarly x+b in S.
           And a+b in S'.
           a+b in S' means a+b in S OR a+b == x.
        """
        # Get candidates a such that x+a in S
        # a must be distinct from x? a is from 1..128.
        # If a == x, then x+a = 2x. If 2x in S, then a=x is a candidate?
        # But a must be in S. x is not in S. So a != x.
        
        A_x = []
        for a in valid_a[x]:
            # Check if x+a is in S
            if (S_mask >> ((x + a) - 1)) & 1:
                A_x.append(a)
        
        # Now check if there are distinct a, b in A_x such that a+b is in S'
        # a+b in S' means (a+b in S) OR (a+b == x)
        
        n = len(A_x)
        for i in range(n):
            a = A_x[i]
            for j in range(i + 1, n):
                b = A_x[j]
                s = a + b
                if s == x:
                    return False
                # Check if s is in S
                if s <= N and (S_mask >> (s - 1)) & 1:
                    return False
        return True

    # Initial solution: Upper half [65..128]
    # Bits 64..127 set
    init_mask = 0
    for v in range(65, 129):
        init_mask |= (1 << (v - 1))
        
    best_mask = init_mask
    best_size = 64
    
    current_mask = init_mask
    current_size = 64
    
    # Local Search
    # We will use a simple hill climbing with random restarts or simulated annealing.
    # Given the speed of can_add, we can do many steps.
    
    # To speed up can_add, we can precompute A_x more efficiently?
    # A_x depends on S_mask.
    
    # Let's run a loop of add/remove steps
    steps = 0
    while time.time() - start_time < DEADLINE:
        # Try to add a random element
        # We can prioritize elements that are "likely" to fit?
        # Or just random.
        
        # Find all elements not in S
        # Instead of building a list, pick a random candidate
        # If we pick x and it's in S, retry?
        # Better: maintain a list of available elements?
        # Or just iterate 1..128 and pick random from those not in mask.
        
        # Efficiently pick a random x not in current_mask
        x = random.randint(1, 128)
        if (current_mask >> (x - 1)) & 1:
            continue
            
        if can_add(x, current_mask):
            current_mask |= (1 << (x - 1))
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            steps += 1
        else:
            # If we can't add, try to remove a random element from S to "unlock" additions
            # Or just restart?
            # Let's try removing a random element
            # Pick random y in S
            # Generate random bit position
            y = random.randint(1, 128)
            if not ((current_mask >> (y - 1)) & 1):
                continue
                
            current_mask &= ~(1 << (y - 1))
            current_size -= 1
            steps += 1
            
            # After removal, try to add some elements immediately
            # This is a "kick"
            for _ in range(10):
                if time.time() - start_time > DEADLINE: break
                x = random.randint(1, 128)
                if (current_mask >> (x - 1)) & 1:
                    continue
                if can_add(x, current_mask):
                    current_mask |= (1 << (x - 1))
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask

    # Convert best_mask to list of ints
    result = []
    for v in range(1, 129):
        if (best_mask >> (v - 1)) & 1:
            result.append(v)
            
    return result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Set: {result[:20]}...{result[-20:]}")
    
    # Verify
    S = set(result)
    valid = True
    for a in result:
        for b in result:
            if a >= b:
                continue
            for c in result:
                if c <= b:
                    continue
                if (a+b) in S and (a+c) in S and (b+c) in S:
                    valid = False
                    print(f"Forbidden triple: {a},{b},{c}")
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
