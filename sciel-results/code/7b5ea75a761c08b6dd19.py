# Inside restart loop:
indices = list(range(num_blocks))
random.shuffle(indices)

while uncovered:
    # ...
    for i in indices:
        gain = (unc & m_list[i]).bit_count()
        # ...
