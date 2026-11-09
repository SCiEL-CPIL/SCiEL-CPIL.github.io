ccw_loads[i, :] = 0
for e in range(n_nodes):
    if cw_loads[i, e] == 0:
        ccw_loads[i, e] = w
