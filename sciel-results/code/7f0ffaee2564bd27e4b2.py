import time
import random

def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin before 90s kill
    
    def is_valid_set(A):
        """Verify that a set satisfies the cardinality-separated subset sum property."""
        if not A:
            return True
        n = len(A)
        # S[k] = set of sums of all k-element subsets
        S = [set() for _ in range(n + 1)]
        S[0] = {0}
        for x in A:
            for k in range(n, 0, -1):
                if S[k-1]:
                    S[k].update(s + x for s in S[k-1])
        # Check: for any i != j, S[i] and S[j] must be disjoint
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if S[i] & S[j]:
                    return False
        return True
    
    def run_greedy(order):
        """
        Greedily build a set from 'order' such that for any two subsets of different
        cardinalities, their sums are distinct.
        
        We maintain S[k] = set of sums of all k-element subsets of the current set.
        When adding x, we check if (S[j-1] + x) intersects S[i] for any i != j.
        """
        S = [set() for _ in range(61)]
        S[0] = {0}
        A = []
        current_size = 0
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            
            valid = True
            # Check validity: for each j from 1 to current_size+1
            # new sums are S[j-1] + x
            # These must not be in S[i] for any i != j
            
            # Precompute forbidden sets for each j
            # For a given j, forbidden values are those in S[i] for i != j
            # We need to check if (S[j-1] + x) has any element in forbidden
            
            for j in range(1, current_size + 2):
                if not S[j-1]:
                    continue
                
                # Build forbidden set: union of S[i] for i in [0, current_size] where i != j
                # But we only need to check intersection with S[j-1] + x
                # So we can check: for each s in S[j-1], is s+x in any S[i] for i != j?
                
                # Optimization: precompute union of all S[i] for i != j
                # But this is expensive. Instead, check each S[i] individually but break early.
                
                for s in S[j-1]:
                    candidate = s + x
                    for i in range(current_size + 1):
                        if i == j:
                            continue
                        if candidate in S[i]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                A.append(x)
                current_size += 1
                # Update S
                for k in range(current_size, 0, -1):
                    S[k].update(s + x for s in S[k-1])
        
        return A
    
    # Try several deterministic orders first
    orders = [
        list(range(1, 257)),  # ascending
        list(range(256, 0, -1)),  # descending
        [256, 1, 255, 2, 254, 3, 253, 4, 252, 5, 251, 6, 250, 7, 249, 8, 248, 9, 247, 10,
         246, 11, 245, 12, 244, 13, 243, 14, 242, 15, 241, 16, 240, 17, 239, 18, 238, 19, 237, 20,
         236, 21, 235, 22, 234, 23, 233, 24, 232, 25, 231, 26, 230, 27, 229, 28, 228, 29, 227, 30,
         226, 31, 225, 32, 224, 33, 223, 34, 222, 35, 221, 36, 220, 37, 219, 38, 218, 39, 217, 40,
         216, 41, 215, 42, 214, 43, 213, 44, 212, 45, 211, 46, 210, 47, 209, 48, 208, 49, 207, 50,
         206, 51, 205, 52, 204, 53, 203, 54, 202, 55, 201, 56, 200, 57, 199, 58, 198, 59, 197, 60,
         196, 61, 195, 62, 194, 63, 193, 64, 192, 65, 191, 66, 190, 67, 189, 68, 188, 69, 187, 70,
         186, 71, 185, 72, 184, 73, 183, 74, 182, 75, 181, 76, 180, 77, 179, 78, 178, 79, 177, 80,
         176, 81, 175, 82, 174, 83, 173, 84, 172, 85, 171, 86, 170, 87, 169, 88, 168, 89, 167, 90,
         166, 91, 165, 92, 164, 93, 163, 94, 162, 95, 161, 96, 160, 97, 159, 98, 158, 99, 157, 100,
         156, 101, 155, 102, 154, 103, 153, 104, 152, 105, 151, 106, 150, 107, 149, 108, 148, 109, 147, 110,
         146, 111, 145, 112, 144, 113, 143, 114, 142, 115, 141, 116, 140, 117, 139, 118, 138, 119, 137, 120,
         136, 121, 135, 122, 134, 123, 133, 124, 132, 125, 131, 126, 130, 127, 129, 128],  # alternating high-low
        [128, 1, 256, 2, 129, 3, 255, 4, 130, 5, 254, 6, 131, 7, 253, 8, 132, 9, 252, 10,
         133, 11, 251, 12, 134, 13, 250, 14, 135, 15, 249, 16, 136, 17, 248, 18, 137, 19, 247, 20,
         138, 21, 246, 22, 139, 23, 245, 24, 140, 25, 244, 26, 141, 27, 243, 28, 142, 29, 242, 30,
         143, 31, 241, 32, 144, 33, 240, 34, 145, 35, 239, 36, 146, 37, 238, 38, 147, 39, 237, 40,
         148, 41, 236, 42, 149, 43, 235, 44, 150, 45, 234, 46, 151, 47, 233, 48, 152, 49, 232, 50,
         153, 51, 231, 52, 154, 53, 230, 54, 155, 55, 229, 56, 156, 57, 228, 58, 157, 59, 227, 60,
         158, 61, 226, 62, 159, 63, 225, 64, 160, 65, 224, 66, 161, 67, 223, 68, 162, 69, 222, 70,
         163, 71, 221, 72, 164, 73, 220, 74, 165, 75, 219, 76, 166, 77, 218, 78, 167, 79, 217, 80,
         168, 81, 216, 82, 169, 83, 215, 84, 170, 85, 214, 86, 171, 87, 213, 88, 172, 89, 212, 90,
         173, 91, 211, 92, 174, 93, 210, 94, 175, 95, 209, 96, 176, 97, 208, 98, 177, 99, 207, 100,
         178, 101, 206, 102, 179, 103, 205, 104, 180, 105, 204, 106, 181, 107, 203, 108, 182, 109, 202, 110,
         183, 111, 201, 112, 184, 113, 200, 114, 185, 115, 199, 116, 186, 117, 198, 118, 187, 119, 197, 120,
         188, 121, 196, 122, 189, 123, 195, 124, 190, 125, 194, 126, 191, 127, 193, 128, 192, 129, 191, 130,
         190, 131, 189, 132, 188, 133, 187, 134, 186, 135, 185, 136, 184, 137, 183, 138, 182, 139, 181, 140,
         180, 141, 179, 142, 178, 143, 177, 144, 176, 145, 175, 146, 174, 147, 173, 148, 172, 149, 171, 150,
         170, 151, 169, 152, 168, 153, 167, 154, 166, 155, 165, 156, 164, 157, 163, 158, 162, 159, 161, 160],  # mid-out
    ]
    
    best_set = []
    for order in orders:
        if time.time() - start_time > deadline:
            break
        candidate = run_greedy(order)
        if len(candidate) > len(best_set):
            best_set = candidate
    
    # Randomized search with bias toward larger numbers
    # Strategy: generate random permutations but with a bias
    for _ in range(100):
        if time.time() - start_time > deadline:
            break
        
        # Create a biased order: mix of random and sorted
        base = list(range(1, 257))
        random.shuffle(base)
        
        # With some probability, sort by value (ascending or descending)
        r = random.random()
        if r < 0.2:
            order = sorted(base)
        elif r < 0.4:
            order = sorted(base, reverse=True)
        else:
            order = base
        
        candidate = run_greedy(order)
        if len(candidate) > len(best_set):
            best_set = candidate
    
    # Final random shuffles
    while time.time() - start_time < deadline:
        order = list(range(1, 257))
        random.shuffle(order)
        candidate = run_greedy(order)
        if len(candidate) > len(best_set):
            best_set = candidate
    
    # Verify the best set is valid
    if best_set and not is_valid_set(best_set):
        # Fallback to a simple valid set
        best_set = [1, 2, 4, 8, 16, 32, 64, 128]
    
    return [int(x) for x in best_set]
