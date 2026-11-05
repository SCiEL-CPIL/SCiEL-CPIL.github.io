import time
import random
import math

def construct():
    N = 512
    # Elements are 1..N, mapped to bits 0..N-1
    # Bit i represents element i+1
    
    # Precompute triples for each element
    # triples_for[x] is a list of integer masks for the other 3 elements
    triples_for = [[] for _ in range(N)]
    
    # Generate all 4-APs
    # a, a+d, a+2d, a+3d
    # 1 <= a, a+3d <= N
    # d >= 1
    
    # We iterate over d and a
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # Convert to 0-indexed bits
            a_bit = 1 << (a - 1)
            b_bit = 1 << (b - 1)
            c_bit = 1 << (c - 1)
            e_bit = 1 << (e - 1)
            
            # For each element in the AP, the other 3 form a triple
            # Element a (index a-1)
            mask_bce = b_bit | c_bit | e_bit
            triples_for[a - 1].append(mask_bce)
            
            # Element b (index b-1)
            mask_ace = a_bit | c_bit | e_bit
            triples_for[b - 1].append(mask_ace)
            
            # Element c (index c-1)
            mask_abe = a_bit | b_bit | e_bit
            triples_for[c - 1].append(mask_abe)
            
            # Element e (index e-1)
            mask_abc = a_bit | b_bit | c_bit
            triples_for[e - 1].append(mask_abc)

    # Initial state: Greedy construction to get a valid starting point
    # This ensures we start with 0 APs
    S = 0
    size = 0
    num_aps = 0
    
    # Simple greedy: add x if it doesn't create an AP
    for x in range(1, N + 1):
        x_idx = x - 1
        x_bit = 1 << x_idx
        # Check if adding x creates any AP
        # An AP is created if any triple in triples_for[x_idx] is already in S
        creates_ap = False
        for mask in triples_for[x_idx]:
            if (S & mask) == mask:
                creates_ap = True
                break
        if not creates_ap:
            S |= x_bit
            size += 1
            # Note: num_aps remains 0 because we only add if no AP is created
            
    best_S = S
    best_size = size
    best_score = size # Since num_aps=0, score is size
    
    start_time = time.time()
    deadline = 50.0 # Leave 10s buffer for 60s limit
    
    T = 10.0
    cooling = 0.99
    
    # Random seed for reproducibility in evaluation? No, random is fine, but let's fix it for stability if needed.
    # random.seed(42) # Optional
    
    while time.time() - start_time < deadline:
        # Pick a random move
        x = random.randint(1, N)
        x_idx = x - 1
        x_bit = 1 << x_idx
        
        if S & x_bit:
            # Remove x
            # Count APs destroyed
            k = 0
            for mask in triples_for[x_idx]:
                if (S & mask) == mask:
                    k += 1
            
            new_num_aps = num_aps - k
            new_size = size - 1
            delta_F = (new_num_aps - new_size) - (num_aps - size)
            # delta_F = -k - (-1) = 1 - k
            
            # Check acceptance
            if delta_F < 0 or random.random() < math.exp(-delta_F / T):
                S ^= x_bit
                num_aps = new_num_aps
                size = new_size
                
                if num_aps == 0 and size > best_size:
                    best_size = size
                    best_S = S
                    best_score = size
        else:
            # Add x
            # Count APs created
            k = 0
            for mask in triples_for[x_idx]:
                if (S & mask) == mask:
                    k += 1
            
            new_num_aps = num_aps + k
            new_size = size + 1
            delta_F = (new_num_aps - new_size) - (num_aps - size)
            # delta_F = k - 1
            
            if delta_F < 0 or random.random() < math.exp(-delta_F / T):
                S |= x_bit
                num_aps = new_num_aps
                size = new_size
                
                if num_aps == 0 and size > best_size:
                    best_size = size
                    best_S = S
                    best_score = size
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Return the best valid set found
    # Convert bit mask to sorted list
    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)
            
    return sorted(result)

# EVOLVE-BLOCK-END
