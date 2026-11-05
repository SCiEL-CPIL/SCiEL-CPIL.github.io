import time
import random

def count_mono_ap(colors):
    """Count monochromatic 4-term arithmetic progressions."""
    n = len(colors)
    count = 0
    # d max is (n-1)//3
    for d in range(1, (n - 1) // 3 + 1):
        # a max is n - 3*d
        for a in range(n - 3 * d):
            if colors[a] == colors[a + d] == colors[a + 2 * d] == colors[a + 3 * d]:
                count += 1
    return count

def thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    return [bin(i).count('1') % 2 for i in range(n)]

def construct():
    n = 256
    colors = thue_morse(n)
    current_score = count_mono_ap(colors)
    
    max_time = 110
    start_time = time.time()
    max_d = (n - 1) // 3
    
    def calc_delta(i, colors):
        c = colors[i]
        new_c = 1 - c
        delta = 0
        
        for d in range(1, max_d + 1):
            # Case 1: i is a
            if i + 3 * d < n:
                if colors[i+d] == c and colors[i+2*d] == c and colors[i+3*d] == c:
                    delta -= 1
                elif colors[i+d] == new_c and colors[i+2*d] == new_c and colors[i+3*d] == new_c:
                    delta += 1

            # Case 2: i is a+d
            if i - d >= 0 and i + 2 * d < n:
                if colors[i-d] == c and colors[i+d] == c and colors[i+2*d] == c:
                    delta -= 1
                elif colors[i-d] == new_c and colors[i+d] == new_c and colors[i+2*d] == new_c:
                    delta += 1

            # Case 3: i is a+2d
            if i - 2 * d >= 0 and i + d < n:
                if colors[i-2*d] == c and colors[i-d] == c and colors[i+d] == c:
                    delta -= 1
                elif colors[i-2*d] == new_c and colors[i-d] == new_c and colors[i+d] == new_c:
                    delta += 1

            # Case 4: i is a+3d
            if i - 3 * d >= 0:
                if colors[i-3*d] == c and colors[i-2*d] == c and colors[i-d] == c:
                    delta -= 1
                elif colors[i-3*d] == new_c and colors[i-2*d] == new_c and colors[i-d] == new_c:
                    delta += 1
                        
        return delta

    # Main loop
    while time.time() - start_time < max_time:
        improved = False
        for i in range(n):
            delta = calc_delta(i, colors)
            if delta < 0:
                colors[i] ^= 1
                current_score += delta
                improved = True
        
        if not improved:
            if time.time() - start_time < max_time - 2:
                num_flips = random.randint(3, 8)
                for _ in range(num_flips):
                    idx = random.randint(0, n - 1)
                    colors[idx] ^= 1
                current_score = count_mono_ap(colors)
            else:
                break
                
    return colors
