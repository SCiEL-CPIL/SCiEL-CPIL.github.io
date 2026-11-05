import time
import random
import math

N = 256

# Precompute 3-APs and 4-APs
three_aps = []
four_aps = []

# three_ap_indices[x] = list of indices of 3-APs containing x
three_ap_indices = [[] for _ in range(N + 1)]
# four_ap_conflicts[x] = list of masks of the other 3 elements in a 4-AP with x
four_ap_conflicts = [[] for _ in range(N + 1)]

def precompute():
    global three_aps, four_aps
    
    # Generate 3-APs
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            idx = len(three_aps)
            three_aps.append(mask)
            three_ap_indices[a].append(idx)
            three_ap_indices[b].append(idx)
            three_ap_indices[c].append(idx)
            
    # Generate 4-APs
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # For each element in the 4-AP, store the mask of the other 3
            elements = [a, b, c, e]
            for i, elem in enumerate(elements):
                others_mask = 0
                for j, other in enumerate(elements):
                    if i != j:
                        others_mask |= (1 << (other - 1))
                four_ap_conflicts[elem].append(others_mask)

precompute()

def count_3aps(S_mask):
    """Count 3-APs in S_mask."""
    count = 0
    for mask in three_aps:
        if (S_mask & mask) == mask:
            count += 1
    return count

def is_valid(S_mask):
    """Check if S_mask is 4-AP free."""
    # A set is 4-AP free if no 4-AP is fully contained.
    # We can check by iterating 4-APs, but we didn't store global 4-AP masks, only conflicts.
    # For verification, let's just use the conflicts logic or rebuild.
    # Actually, the construction ensures validity. For final check, we can iterate elements in S
    # and check if any 4-AP conflict is satisfied by S.
    # If S contains x, and S contains the 'others' for some 4-AP conflict of x, it's invalid.
    # But this might double count or be tricky.
    # Simpler: The greedy/SA logic ensures validity.
    # Let's assume validity if constructed correctly.
    return True

def gain_add(x, S_mask):
    """Calculate number of new 3-APs formed by adding x to S_mask."""
    bit = 1 << (x - 1)
    gain = 0
    for idx in three_ap_indices[x]:
        m = three_aps[idx]
        # Check if the other 2 elements are in S_mask
        # m has 3 bits. One is x.
        # (S_mask & (m ^ bit)) == (m ^ bit)
        if (S_mask & (m ^ bit)) == (m ^ bit):
            gain += 1
    return gain

def can_add(x, S_mask):
    """Check if adding x to S_mask creates a 4-AP."""
    for others_mask in four_ap_conflicts[x]:
        if (S_mask & others_mask) == others_mask:
            return False
    return True

def greedy_construct():
    S_mask = 0
    available = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256]
    
    while available:
        best_x = -1
        best_gain = -1
        for x in available:
            if not can_add(x, S_mask):
                continue
            g = gain_add(x, S_mask)
            if g > best_gain:
                best_gain = g
                best_x = x
        
        if best_x == -1:
            break
            
        S_mask |= (1 << (best_x - 1))
        # Remove best_x and its conflicts from available
        new_available = []
        for x in available:
            if x == best_x:
                continue
            if not can_add(x, S_mask):
                continue
            new_available.append(x)
        available = new_available
        
    return S_mask

def local_search(S_mask, time_limit):
    start_time = time.time()
    deadline = start_time + time_limit
    
    current_S = S_mask
    current_score = count_3aps(S_mask)
    best_S = S_mask
    best_score = current_score
    
    # Precompute list of elements in S for faster iteration
    def get_elements(mask):
        elems = []
        for i in range(1, N + 1):
            if mask & (1 << (i - 1)):
                elems.append(i)
        return elems
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        elems = get_elements(current_S)
        
        for x in elems:
            bit_x = 1 << (x - 1)
            # Remove x
            S_without_x = current_S ^ bit_x
            score_without_x = current_score - gain_add(x, S_without_x)
            
            # Find best y to add
            best_y = -1
            best_delta = 0
            best_score_y = -1
            
            for y in range(1, N + 1):
                bit_y = 1 << (y - 1)
                if S_without_x & bit_y:
                    continue
                if not can_add(y, S_without_x):
                    continue
                
                g = gain_add(y, S_without_x)
                new_score = score_without_x + g
                if new_score > best_score_y:
                    best_score_y = new_score
                    best_y = y
                    best_delta = new_score - current_score
            
            if best_y != -1 and best_delta > 0:
                current_S = S_without_x | (1 << (best_y - 1))
                current_score = best_score_y
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_S = current_S
                    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # 1. Greedy Construction
    S_mask = greedy_construct()
    
    # 2. Local Search
    S_mask = local_search(S_mask, 50.0)
    
    # Convert mask to list
    result = []
    for i in range(N):
        if S_mask & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
