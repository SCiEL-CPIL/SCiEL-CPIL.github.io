import time

def construct():
    """
    Uses greedy minimal extension to find 12 distinct positive integers
    with unique subset sums, minimizing the largest integer.
    
    At each step, select the smallest unused positive integer that does not
    create a collision with any existing subset sum.
    """
    n = 12
    deadline = time.time() + 85  # Safety margin before 90s kill

    elements = []
    subset_sums = {0}
    used = set()

    for i in range(n):
        if time.time() > deadline:
            break
        candidate = 1
        while True:
            if time.time() > deadline:
                break
            if candidate not in used:
                # Check if adding candidate creates any subset sum collision
                valid = True
                for s in subset_sums:
                    if (s + candidate) in subset_sums:
                        valid = False
                        break
                if valid:
                    elements.append(candidate)
                    used.add(candidate)
                    new_sums = {s + candidate for s in subset_sums}
                    subset_sums = subset_sums | new_sums
                    break
            candidate += 1

    return elements
