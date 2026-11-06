if delta_energy < 0:
    accept = True
else:
    # delta_energy >= 0
    if T == 0 or delta_energy / T > 50: # exp(-50) is negligible
        accept = False
    else:
        accept = random.random() < math.exp(-delta_energy / T)
