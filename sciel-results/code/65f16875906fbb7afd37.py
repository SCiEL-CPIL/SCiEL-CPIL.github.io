import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    n = 16
    edges = []

    # K_{8,8} is the unique 16-vertex graph with 64 edges, diameter 2,
    # and edge-criticality (every edge essential for diameter-2 property).
    # Using XOR partition for a structurally distinct construction path.
    for u in range(n):
        if time.time() > deadline:
            break
        for v in range(u + 1, n):
            if (u ^ v) >= 8:
                edges.append([u, v])

    return edges
