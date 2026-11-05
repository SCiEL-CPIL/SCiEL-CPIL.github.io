"""Program for the suite task: maximize subset of [1,100] with no subset summing to 200."""
# EVOLVE-BLOCK-START
def construct():
    # Greedy approach: iterate from 1 to 100, include a number if it doesn't
    # create a subset sum of 200 with the currently selected set.
    # We track all reachable sums using the selected elements.
    
    selected = []
    reachable_sums = {0}
    
    for x in range(1, 101):
        # Check if adding x would create a sum of 200
        # This happens if there exists s in reachable_sums such that s + x == 200
        # i.e., 200 - x is in reachable_sums
        if (200 - x) not in reachable_sums:
            # Adding x is safe; update reachable sums
            new_sums = set()
            for s in reachable_sums:
                ns = s + x
                if ns <= 200:
                    new_sums.add(ns)
            reachable_sums |= new_sums
            selected.append(x)
    
    # Ensure nonempty
    if not selected:
        return [1]
    
    return selected
# EVOLVE-BLOCK-END
