    strategies = [
        ('min', 0),
        ('max', 0),
        ('asc', 0),
        ('desc', 0)
    ]
    # Add 10 random restarts
    for i in range(10):
        strategies.append(('random', i + 1))
