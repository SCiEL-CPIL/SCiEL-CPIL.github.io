"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85  # 85s safety margin (90s kill limit)
    
    def is_mean_of_subset_bitset(element, others):
        """Check if element is the arithmetic mean of any subset of others with size >= 2.
        Uses bitset DP for speed. Returns True if element is the mean of some subset."""
        n = len(others)
        if n < 2:
            return False
        target = element
        
        # Sort others for quick bound checks
        sorted_others = sorted(others)
        prefix_sums = [0]
        for v in sorted_others:
            prefix_sums.append(prefix_sums[-1] + v)
        
        for k in range(2, n + 1):
            target_sum = k * target
            
            # Quick bounds check
            min_possible = prefix_sums[k]
            max_possible = prefix_sums[-1] - (prefix_sums[-1 - k] if k < n else 0)
            if target_sum < min_possible or target_sum > max_possible:
                continue
            
            # Bitset DP: dp[j] is a bitmask where bit s is set if sum s is achievable with j elements
            # Max possible sum is sum of all others
            max_sum = prefix_sums[-1]
            # We only care about sums up to target_sum
            dp = [0] * (k + 1)
            dp[0] = 1  # sum 0 with 0 elements
            
            for val in sorted_others:
                for j in range(k, 0, -1):
                    if dp[j - 1]:
                        # Shift left by val, mask to target_sum bits
                        dp[j] |= (dp[j - 1] << val)
                        # Early exit: check if target_sum bit is set
                        if (dp[j] >> target_sum) & 1:
                            return True
            
            if (dp[k] >> target_sum) & 1:
                return True
        
        return False
    
    def can_add_element_fast(element, current_set):
        """Check if element can be added to current_set without violating the constraint.
        Returns True if element can be safely added."""
        # Check 1: is element the mean of any subset of current_set?
        if current_set and is_mean_of_subset_bitset(element, current_set):
            return False
        
        # Check 2: for each a in current_set, does adding element create a new mean violation?
        # a is mean of subset of (current_set - {a}) + {element} that includes element
        # => exists subset T of current_set - {a} with |T| >= 1 such that sum(T) = (|T|+1)*a - element
        for a in current_set:
            others_without_a = [x for x in current_set if x != a]
            n = len(others_without_a)
            if n == 0:
                continue
            
            # Sort for bounds
            sorted_others = sorted(others_without_a)
            prefix_sums = [0]
            for v in sorted_others:
                prefix_sums.append(prefix_sums[-1] + v)
            
            for j in range(1, n + 1):
                target_sum = (j + 1) * a - element
                if target_sum <= 0:
                    continue
                
                # Bounds check
                min_possible = prefix_sums[j]
                max_possible = prefix_sums[-1] - (prefix_sums[-1 - j] if j < n else 0)
                if target_sum < min_possible or target_sum > max_possible:
                    continue
                
                # Bitset DP check
                dp = [0] * (j + 1)
                dp[0] = 1
                
                for val in sorted_others:
                    for jj in range(j, 0, -1):
                        if dp[jj - 1]:
                            dp[jj] |= (dp[jj - 1] << val)
                            if (dp[jj] >> target_sum) & 1:
                                return False
                
                if (dp[j] >> target_sum) & 1:
                    return False
        
        return True
    
    # Strategy: build a large valid set using multiple orderings
    best_set = []
    
    # Define multiple orderings to try
    all_nums = list(range(1, 513))
    powers_of_2 = [2**i for i in range(9) if 2**i <= 512]
    non_powers = [x for x in range(1, 513) if x not in set(powers_of_2)]
    
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True
    
    primes = [x for x in range(2, 513) if is_prime(x)]
    non_primes = [x for x in range(1, 513) if not is_prime(x)]
    
    # Generate more diverse orderings
    orderings = [
        list(range(1, 513)),           # ascending
        list(range(512, 0, -1)),       # descending
        # Powers of 2 first (known to work well)
        powers_of_2 + non_powers,
        # Odds first
        [x for x in range(1, 513) if x % 2 == 1] + [x for x in range(1, 513) if x % 2 == 0],
        # Evens first
        [x for x in range(1, 513) if x % 2 == 0] + [x for x in range(1, 513) if x % 2 == 1],
        # Non-multiples of 3 first
        [x for x in range(1, 513) if x % 3 != 0] + [x for x in range(1, 513) if x % 3 == 0],
        # Large numbers first
        list(range(512, 256, -1)) + list(range(256, 0, -1)),
        # Primes first (heuristic)
        primes + non_primes,
        # Small numbers first
        list(range(1, 257)) + list(range(512, 256, -1)),
        # Alternating small/large
        [1, 512, 2, 511, 3, 510, 4, 509, 5, 508, 6, 507, 7, 506, 8, 505, 9, 504, 10, 503, 11, 502, 12, 501, 13, 500, 14, 499, 15, 498, 16, 497, 17, 496, 18, 495, 19, 494, 20, 493, 21, 492, 22, 491, 23, 490, 24, 489, 25, 488, 26, 487, 27, 486, 28, 485, 29, 484, 30, 483, 31, 482, 32, 481, 33, 480, 34, 479, 35, 478, 36, 477, 37, 476, 38, 475, 39, 474, 40, 473, 41, 472, 42, 471, 43, 470, 44, 469, 45, 468, 46, 467, 47, 466, 48, 465, 49, 464, 50, 463, 51, 462, 52, 461, 53, 460, 54, 459, 55, 458, 56, 457, 57, 456, 58, 455, 59, 454, 60, 453, 61, 452, 62, 451, 63, 450, 64, 449, 65, 448, 66, 447, 67, 446, 68, 445, 69, 444, 70, 443, 71, 442, 72, 441, 73, 440, 74, 439, 75, 438, 76, 437, 77, 436, 78, 435, 79, 434, 80, 433, 81, 432, 82, 431, 83, 430, 84, 429, 85, 428, 86, 427, 87, 426, 88, 425, 89, 424, 90, 423, 91, 422, 92, 421, 93, 420, 94, 419, 95, 418, 96, 417, 97, 416, 98, 415, 99, 414, 100, 413, 101, 412, 102, 411, 103, 410, 104, 409, 105, 408, 106, 407, 107, 406, 108, 405, 109, 404, 110, 403, 111, 402, 112, 401, 113, 400, 114, 399, 115, 398, 116, 397, 117, 396, 118, 395, 119, 394, 120, 393, 121, 392, 122, 391, 123, 390, 124, 389, 125, 388, 126, 387, 127, 386, 128, 385, 129, 384, 130, 383, 131, 382, 132, 381, 133, 380, 134, 379, 135, 378, 136, 377, 137, 376, 138, 375, 139, 374, 140, 373, 141, 372, 142, 371, 143, 370, 144, 369, 145, 368, 146, 367, 147, 366, 148, 365, 149, 364, 150, 363, 151, 362, 152, 361, 153, 360, 154, 359, 155, 358, 156, 357, 157, 356, 158, 355, 159, 354, 160, 353, 161, 352, 162, 351, 163, 350, 164, 349, 165, 348, 166, 347, 167, 346, 168, 345, 169, 344, 170, 343, 171, 342, 172, 341, 173, 340, 174, 339, 175, 338, 176, 337, 177, 336, 178, 335, 179, 334, 180, 333, 181, 332, 182, 331, 183, 330, 184, 329, 185, 328, 186, 327, 187, 326, 188, 325, 189, 324, 190, 323, 191, 322, 192, 321, 193, 320, 194, 319, 195, 318, 196, 317, 197, 316, 198, 315, 199, 314, 200, 313, 201, 312, 202, 311, 203, 310, 204, 309, 205, 308, 206, 307, 207, 306, 208, 305, 209, 304, 210, 303, 211, 302, 212, 301, 213, 300, 214, 299, 215, 298, 216, 297, 217, 296, 218, 295, 219, 294, 220, 293, 221, 292, 222, 291, 223, 290, 224, 289, 225, 288, 226, 287, 227, 286, 228, 285, 229, 284, 230, 283, 231, 282, 232, 281, 233, 280, 234, 279, 235, 278, 236, 277, 237, 276, 238, 275, 239, 274, 240, 273, 241, 272, 242, 271, 243, 270, 244, 269, 245, 268, 246, 267, 247, 266, 248, 265, 249, 264, 250, 263, 251, 262, 252, 261, 253, 260, 254, 259, 255, 258, 256, 257],
    ]
    
    # Add a few randomized orderings for diversity
    random.seed(42)
    for _ in range(3):
        shuffled = all_nums[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)
    
    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        
        selected = []
        selected_set = set()
        
        for x in ordering:
            if time.time() > deadline - 10:
                break
            if x in selected_set:
                continue
            
            if can_add_element_fast(x, selected):
                selected.append(x)
                selected_set.add(x)
        
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Local improvement: try to add more elements
    if best_set and time.time() < deadline - 15:
        improved = best_set[:]
        improved_set = set(improved)
        remaining = [x for x in range(1, 513) if x not in improved_set]
        
        # Try adding remaining elements
        for x in remaining:
            if time.time() > deadline - 10:
                break
            if can_add_element_fast(x, improved):
                improved.append(x)
                improved_set.add(x)
        
        if len(improved) > len(best_set):
            best_set = improved[:]
    
    # Try swapping: remove one element, try to add multiple others
    if best_set and time.time() < deadline - 20:
        for i in range(len(best_set)):
            if time.time() > deadline - 10:
                break
            without_i = [x for j, x in enumerate(best_set) if j != i]
            without_i_set = set(without_i)
            remaining2 = [x for x in range(1, 513) if x not in without_i_set]
            
            # Try to add as many as possible from remaining2
            new_set = without_i[:]
            new_set_set = without_i_set.copy()
            for x in remaining2:
                if time.time() > deadline - 10:
                    break
                if can_add_element_fast(x, new_set):
                    new_set.append(x)
                    new_set_set.add(x)
            
            if len(new_set) > len(best_set):
                best_set = new_set[:]
                break
    
    return best_set
# EVOLVE-BLOCK-END
